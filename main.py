import streamlit as st
from pydantic import BaseModel, EmailStr, Field
from typing import List, Literal
import cohere
import os
import random
import json
from datetime import datetime
from dotenv import load_dotenv
import csv
from io import StringIO
from gtts import gTTS
from io import BytesIO
from pathlib import Path

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
    interest: str = Field(..., min_length=1, max_length=100)
    tone: Literal["formal", "casual", "enthusiastic", "neutral"] = "neutral"
    language: str = Field(..., min_length=1, max_length=200)

class EmailVariant(BaseModel):
    subject: str
    body: str
    call_to_action: str
    sub_variants: List[str] = []
    suggested_send_time: str  # List of alternative subject ideas

class Email(BaseModel):
    variants: List[EmailVariant]

class Campaign(BaseModel):
    account_name: str
    emails: List[Email]

class CampaignRequest(BaseModel):
    accounts: List[Account] = Field(..., min_items=1, max_items=10)
    number_of_emails: int = Field(..., gt=0, le=10)

# Dependency for Cohere client
def get_cohere_client():
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise ValueError("COHERE_API_KEY environment variable is not set")
    return cohere.Client(api_key)

def generate_email_content(client: cohere.Client, account: Account, email_number: int, total_emails: int, tone: str) -> List[EmailVariant]:
    if tone not in ["formal", "casual", "enthusiastic", "neutral"]:
        raise ValueError("Invalid tone provided. Must be one of: formal, casual, enthusiastic, neutral.")

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
        raise ValueError("Invalid JSON response from Cohere")

    sub_variants = email_data.get("subject", [])
    salutation = f"Best regards,The {account.account_name} Team"

    send_times = {
        "morning": "8 AM - 10 AM",
        "afternoon": "1 PM - 3 PM",
        "evening": "6 PM - 8 PM",
    }

    if account.industry.lower() in ["technology", "software"]:
        recommended_send_time = send_times["morning"]
    elif account.industry.lower() in ["retail", "e-commerce"]:
        recommended_send_time = send_times["afternoon"]
    else:
        recommended_send_time = send_times["evening"]

    return [
        EmailVariant(
            subject=sub_variants[0],
            body=email_data["body"].replace("\n", ""),
            call_to_action=email_data["call_to_action"].replace("\n", "") + salutation.replace("\n", ""),
            sub_variants=sub_variants,
            suggested_send_time=recommended_send_time
        )
    ]

def generate_campaign(client: cohere.Client, account: Account, number_of_emails: int) -> Campaign:
    emails = []
    for contact in account.contacts:
        contact.group = random.choice(["A", "B"])

    for i in range(number_of_emails):
        tone = account.tone if account.tone else "neutral"
        email_variants = generate_email_content(client, account, i + 1, number_of_emails, tone)
        emails.append(Email(variants=email_variants))

    return Campaign(account_name=account.account_name, emails=emails)

# Streamlit UI

st.title("Email Drip Campaign Generator")

api_key = os.getenv("COHERE_API_KEY")
if not api_key:
    st.error("COHERE_API_KEY environment variable is not set.")
else:
    # Form to input account details
    with st.form(key='account_form'):
        account_name = st.text_input("Account Name")
        industry = st.text_input("Industry")
        pain_points = st.text_area("Pain Points (comma separated)").split(",")
        contacts = []
        num_contacts = st.number_input("Number of Contacts", min_value=1, max_value=5)
        for i in range(num_contacts):
            name = st.text_input(f"Contact {i+1} Name")
            email = st.text_input(f"Contact {i+1} Email")
            job_title = st.text_input(f"Contact {i+1} Job Title")
            contacts.append(Contact(name=name, email=email, job_title=job_title))
        
        campaign_objective = st.selectbox("Campaign Objective", ["awareness", "nurturing", "upselling"])
        interest = st.text_input("Interest")
        tone = st.selectbox("Tone", ["formal", "casual", "enthusiastic", "neutral"])
        language = st.text_input("Language")

        num_emails = st.number_input("Number of Emails", min_value=1, max_value=10)

        submit_button = st.form_submit_button("Generate Campaign")
    
    if submit_button:
        if api_key:
            client = get_cohere_client()
            account = Account(
                account_name=account_name,
                industry=industry,
                pain_points=pain_points,
                contacts=contacts,
                campaign_objective=campaign_objective,
                interest=interest,
                tone=tone,
                language=language
            )

            try:
                campaign = generate_campaign(client, account, num_emails)
                st.success("Campaign generated successfully!")

                # Display generated emails
                for email in campaign.emails:
                    for variant in email.variants:
                        st.write(f"Subject: {variant.subject}")
                        st.write(f"Body: {variant.body}")
                        st.write(f"Call to Action: {variant.call_to_action}")
                        st.write(f"Suggested Send Time: {variant.suggested_send_time}")
    
    # Export campaign to CSV
    export_button = st.button("Export Campaign as CSV")
    if export_button:
        # Prepare CSV
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(['Account Name', 'Email Number', 'Variant', 'Subject', 'Sub-Variants', 'Body', 'Call to Action', 'Recommended Send Time'])

        for email in campaign.emails:
            for variant in email.variants:
                writer.writerow([campaign.account_name, f"Email {i + 1}", f"Variant {j + 1}", variant.subject, "; ".join(variant.sub_variants), variant.body, variant.call_to_action, variant.suggested_send_time])

        output.seek(0)
        st.download_button(
            label="Download CSV",
            data=output.getvalue(),
            file_name=f"campaign_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    # TTS generation
    tts_button = st.button("Generate Text to Speech for Email")
    if tts_button:
        email_body = "This is a test body text for TTS."  # Replace with email body of choice
        tts = gTTS(email_body)
        audio = BytesIO()
        tts.save(audio)
        audio.seek(0)
        st.audio(audio)
