from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, EmailStr, Field
from typing import List, Literal
import cohere
import os
import random
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import csv
from io import StringIO
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from gtts import gTTS
from io import BytesIO

# Load environment variables
env_path = Path('.') / 'variables.env'
load_dotenv(dotenv_path=env_path)

# Input Models with new fields for A/B testing
class Contact(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    job_title: str = Field(..., min_length=1, max_length=100)
    group: Literal["A", "B"] = "A"  # Default group for A/B testing

class Account(BaseModel):
    account_name: str = Field(..., min_length=1, max_length=200)
    industry: str = Field(..., min_length=1, max_length=100)
    pain_points: List[str] = Field(..., min_items=1, max_items=5)
    contacts: List[Contact] = Field(..., min_items=1)
    campaign_objective: Literal["awareness", "nurturing", "upselling"]

    # New fields for interest, tone, and language
    interest: str = Field(..., min_length=1, max_length=100)
    tone: Literal["formal", "casual", "enthusiastic", "neutral"] = "neutral"
    language: str = Field(..., min_length=1, max_length=200)

class EmailVariant(BaseModel):
    subject: str
    body: str
    call_to_action: str
    sub_variants: List[str] = []
    suggested_send_time: str    # List of alternative subject ideas

class Email(BaseModel):
    variants: List[EmailVariant]

class Campaign(BaseModel):
    account_name: str
    emails: List[Email]

class CampaignRequest(BaseModel):
    accounts: List[Account] = Field(..., min_items=1, max_items=10)
    number_of_emails: int = Field(..., gt=0, le=10)

class CampaignResponse(BaseModel):
    campaigns: List[Campaign]

# Application lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.getenv("COHERE_API_KEY"):
        raise ValueError("COHERE_API_KEY environment variable is not set")
    yield

app = FastAPI(
    title="Email Drip Campaign API by Error Pointers",
    description="Generate personalized email campaigns ",
    version="1.0.0",
    lifespan=lifespan,
)

# Dependency for Cohere client
def get_cohere_client():
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="COHERE_API key not found")
    return cohere.Client(api_key)

def generate_email_content(client: cohere.Client, account: Account, email_number: int, total_emails: int, tone: str) -> List[EmailVariant]:
    if tone not in ["formal", "casual", "enthusiastic", "neutral"]:
        raise HTTPException(status_code=400, detail="Invalid tone provided. Must be one of: formal, casual, enthusiastic, neutral.")

    prompt = f"""
    Create a personalized email for the following business account:
    Company: {account.account_name}
    Industry: {account.industry}
    Pain Points: {', '.join(account.pain_points)}
    Campaign Stage: Email {email_number} of {total_emails}
    Campaign Objective: {account.campaign_objective}
    Recipient Job Title: {account.contacts[0].job_title}
    Interest: {account.interest}
    Tone: {tone}
    Language: {account.language}

    Generate a catchy and engaging subject line, personalized for the account and campaign objective. Please generate **three distinct subject lines**.

    Then, write the email body content with the following structure:
    1. An engaging email body personalized to the pain points and interest of the account
    2. A clear call-to-action encouraging the recipient to take the next step.
    3. Ensure the body is cohesive and flows well with the subject.

    Format the response as valid JSON with keys: "subject", "body", "call_to_action"
    """
    response = client.generate(
        model="command-xlarge-nightly",
        prompt=prompt,
        max_tokens=400,
        temperature=0.7,
    )

    response_text = response.generations[0].text.strip()

    try:
        email_data = json.loads(response_text)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON response from Cohere")

    sub_variants = email_data.get("subject", [])
    salutation = f"Best regards,The {account.account_name} Team"

    # Define recommended send time based on industry or general rules
    send_times = {
        "morning": "8 AM - 10 AM",
        "afternoon": "1 PM - 3 PM",
        "evening": "6 PM - 8 PM",
    }

    # Example rules: Adjust based on account's industry or type of campaign
    if account.industry.lower() in ["technology", "software"]:
        recommended_send_time = send_times["morning"]  # Morning works well for tech emails
    elif account.industry.lower() in ["retail", "e-commerce"]:
        recommended_send_time = send_times["afternoon"]  # Afternoon may work best for retail
    else:
        recommended_send_time = send_times["evening"]  # Evening is generally safe for other industries

    return [
        EmailVariant(
            subject=sub_variants[0],
            body=email_data["body"].replace("\n", ""),
            call_to_action=email_data["call_to_action"].replace("\n", "") + salutation.replace("\n", ""),
            sub_variants=sub_variants,
            suggested_send_time=recommended_send_time  # Add suggested send time
        )
    ]

def generate_campaign(client: cohere.Client, account: Account, number_of_emails: int) -> Campaign:
    emails = []
    for contact in account.contacts:
        contact.group = random.choice(["A", "B"])  # Assign random group for A/B testing

    for i in range(number_of_emails):
        tone = account.tone if account.tone else "neutral"
        email_variants = generate_email_content(client, account, i + 1, number_of_emails, tone)
        emails.append(Email(variants=email_variants))

    return Campaign(account_name=account.account_name, emails=emails)

@app.post(
    "/generate-campaigns/",
    response_model=CampaignResponse,
    summary="Generate email campaigns",
    response_description="Generated email campaigns for the provided accounts"
)
def generate_campaigns(
    request: CampaignRequest,
    client: cohere.Client = Depends(get_cohere_client)
) -> CampaignResponse:
    try:
        campaigns = [
            generate_campaign(client, account, request.number_of_emails)
            for account in request.accounts
        ]
        return CampaignResponse(campaigns=campaigns)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post(
    "/export-campaigns-csv/",
    summary="Export campaigns as CSV",
    response_description="CSV file containing all generated campaigns"
)
def export_campaigns_csv(
    request: CampaignRequest,
    client: cohere.Client = Depends(get_cohere_client)
):
    campaigns_response = generate_campaigns(request, client)

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Account Name', 'Email Number', 'Variant', 'Subject', 'Sub-Variants', 'Body', 'Call to Action', 'Recommended Send Time'])

    for campaign in campaigns_response.campaigns:
        for email_idx, email in enumerate(campaign.emails, 1):
            for variant_idx, variant in enumerate(email.variants, 1):
                writer.writerow([
                    campaign.account_name,
                    f"Email {email_idx}",
                    f"Variant {variant_idx}",
                    variant.subject,
                    "; ".join(variant.sub_variants),
                    variant.body,
                    variant.call_to_action,
                    variant.suggested_send_time  # Use the correct attribute
                ])

    output.seek(0)
    filename = f"campaigns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


def generate_tts_from_email(email_body: str, language: str = "en",speed: float = 1.5) -> StreamingResponse:
    try:
        email_body = email_body.replace("\n", "")
        tts = gTTS(text=email_body, lang=language, slow=False)
        audio_file = BytesIO()
        tts.write_to_fp(audio_file)  # Use `write_to_fp` for BytesIO
        audio_file.seek(0)

        return StreamingResponse(
            audio_file,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=email_audio.mp3"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")

@app.post("/generate-email-audio/")
def generate_email_audio(email_body: str, language: str = "en"):
    return generate_tts_from_email(email_body=email_body, language=language)

