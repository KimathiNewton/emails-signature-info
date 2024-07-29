# Email Signature Extraction and Structuring with LLMs

## Overview
This project aims to evaluate and compare two different large language models (LLMs) for the task of extracting email signature information and structuring it into a JSON format. The LLMs evaluated are Mistral-large-latest and Open-mistral-nemo . 
The goal of this project is to:
* Create prompts that can extract email signature information.
* Structure the extracted information into a predefined JSON format.
* Generate test cases to cover a range of email scenarios.
* Evaluate and iterate on prompts to improve extraction accuracy.
* Compare the performance of two LLMs: mistral-large-latest and open-mistral-nemo

# Evaluation and Results
## Cosine Similarity
The evaluation is based on cosine similarity between the extracted data and reference signatures i.e the expected output. 
Cosine similarity is used in this context to measure the semantic similarity between the extracted email signature and the expected (ground truth) signature.
It compares the extracted email signatures with the expected outputs. A higher cosine similarity indicates a better match between the extracted and expected information.
## How it works:
Text Conversion: Both the extracted signature and the expected signature are converted into numerical representations (vectors).The sentence_transformers library is used to convert text into numerical embeddings.
Vector Calculation: These numerical representations are then converted into vectors in a high-dimensional space.The cosine of the angle between these two vectors is computed. A value closer to 1 indicates a higher similarity between the two signatures, while a value closer to 0 indicates lower similarity.
Essentially, a higher cosine similarity score means the LLM did a better job of extracting the correct information from the email.

## Initial Prompt
The initial prompt was designed to extract key information from email signatures and format it into a structured JSON output. The prompt template included placeholders for common signature elements such as the name, email, phone number, job title, company, address, website, and social media links. The prompt also specified that fields not present in the email should be omitted from the JSON output.
I formulated this prompt as it provided a clear structure for the LLM to extract signature information. It outlined the desired JSON output format and instructed the model to omit missing fields.
```
def build_prompt():
    prompt_template = """
    Extract the signature information from the following email content.

    Return the information in JSON format matching the following structure:

    {{
      "name": "Full Name",
      "email": "email@example.com",
      "phone": "Phone number",
      "job_title": "Job Title",
      "company": "Company Name",
      "address": "Full Address",
      "website": "Website URL",
      "social_media": {{
        "linkedin": "LinkedIn URL",
        "twitter": "Twitter handle",
        // other social media
      }}
    }}

    If a field is not present in the email, omit it from the JSON.

    Email content:{email_content}

    Extracted signature information:
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["email_content"]
    )
    return prompt
```
# Test Cases
The test cases included various scenarios to evaluate the prompt's effectiveness in different contexts. These included:
* Basic Signature: Emails with complete signature details (name, title, company, etc.).
* Partial Signature: Emails with incomplete signatures, missing one or more elements.
* No Signature: Emails that do not contain any signature information.
* Signatures: Emails with additional formatting or multiple contact details.
* Nested Signatures: Emails with signatures embedded in quoted text from previous conversations.
The rationale for selecting these test cases was to ensure that the prompt could handle a wide variety of real-world email scenarios, from simple to complex. This helped in assessing the robustness and adaptability of the language models.



# Results

## mistral-large-latest
The results with the initial prompt:
The mean score improved from 0.976 to 0.991, indicating that the overall accuracy of the extracted signatures improved with the refined prompt.

Reduced Variability: The standard deviation decreased from 0.092 to 0.019, showing that the performance became more consistent after prompt improvement.

Higher Minimum Score: The minimum score increased from 0.368 to 0.928, which means the lowest performing cases improved significantly.

The improvements to the prompt, including additional examples, have significantly enhanced the performance of the model. The reduced variability and increased mean score suggest that the refined prompt leads to more accurate and consistent extraction of signature information.

open-mistral-nemo
The mean score improved from 0.863 to 0.882, indicating an enhancement in overall performance after prompt refinement.

Increased Variability: The standard deviation increased from 0.213 to 0.229, suggesting that while the average performance improved, there was more variability in the results.

After adding examples to the prompt, the open-mistral-nemo model showed improved average performance and higher minimum scores. However, the increase in standard deviation suggests more variability in the results. This could indicate that while some cases improved, others might have experienced less consistency.

Conclusion
Mistral-large-latest showed significant improvement after prompt iteration, with mean cosine similarity increasing from 0.976 to 0.991 and reduced variability.

Open-mistral-nemo also showed improvement, with mean cosine similarity increasing from 0.863 to 0.882, but with increased variability.

Both models maintained perfect scores for some cases before and after prompt iteration.