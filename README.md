# Email Signature Extraction and Structuring with LLMs

# Overview
This project aims to evaluate and compare the performance of two large language models (LLMs) in the task of extracting email signature information and formatting it into a structured JSON format. The selected LLMs for this task are the Mistral-Large-Latest model and the Open-Mistral-Nemo model. The project involves designing and iterating prompts, generating and evaluating test cases, and analyzing the models' performance based on the extracted information's accuracy and consistency.

## Objective
The primary objective of this project is to develop a reliable method for extracting structured information from email signatures using LLMs. This involves:

1. Prompt Engineering: Creating and refining prompts that guide the LLMs to extract relevant information accurately.
2. Test Case Generation: Designing a comprehensive set of test cases that represent various real-world email scenarios, including simple, partial, complex, and nested signatures.
3. Model Evaluation: Comparing the performance of the two LLMs based on their ability to correctly identify and format email signature information.

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

# Evaluation Metrics
The model's performance was evaluated using cosine similarity between the extracted signature and the expected ground truth output. Cosine similarity measures the similarity between two vectors, with a higher value indicating a better alignment between the predicted and actual signatures.
## Cosine Similarity
The evaluation is based on cosine similarity between the extracted data and reference signatures i.e the expected output. 
Cosine similarity is used in this context to measure the semantic similarity between the extracted email signature and the expected (ground truth) signature.
It compares the extracted email signatures with the expected outputs. A higher cosine similarity indicates a better match between the extracted and expected information.
## How it works:
Text Conversion: Both the extracted signature and the expected signature are converted into numerical representations (vectors).The sentence_transformers library is used to convert text into numerical embeddings.

Vector Calculation: These numerical representations are then converted into vectors in a high-dimensional space.The cosine of the angle between these two vectors is computed. A value closer to 1 indicates a higher similarity between the two signatures, while a value closer to 0 indicates lower similarity.
Essentially, a higher cosine similarity score means the LLM did a better job of extracting the correct information from the email.


## Results For Mistral-large-latest Model in the Initial Prompt:

* Mean cosine similarity: 0.975759
* Standard deviation: 0.091883
* Minimum cosine similarity: 0.367639
* Maximum cosine similarity: 1.000000

## Results For Open-mistral-nemo Model in the Initial Prompt:

* Mean cosine similarity: 0.863052
* Standard deviation: 0.212798
* Minimum cosine similarity: 0.189032
* Maximum cosine similarity: 1.000000

Based on these results, Mistral-large-latest demonstrated superior performance with a higher average cosine similarity and lower standard deviation, indicating greater accuracy and consistency.

# Prompt Iteration
The initial prompt was refined to provide examples and further clarify the expected output format. The updated prompt included:

* Specific examples of email content and corresponding extracted signatures.
* Clarification on how to handle missing information in the signature, of which it should be ommited from the output.

The addition of examples aimed to provide the model with a clearer understanding of the expected output structure, improving the accuracy of the extracted information, especially in more complex cases.
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

    Examples:
    Email content: "Best, John Doe"
    Extracted signature information: {{
      "name": "John Doe"
    }}

    Email content: "Best regards, Michael Brown Project Manager XYZ Solutions michael.brown@xyzsolutions.com"
    Extracted signature information: {{
      "name": "Michael Brown",
      "email": "michael.brown@xyzsolutions.com",
      "job_title": "Project Manager",
      "company": "XYZ Solutions"
    }}

    Email content: "Cheers, Kevin Lee Senior Developer (555) 987-6543"
    Extracted signature information: {{
      "name": "Kevin Lee",
      "job_title": "Senior Developer",
      "phone": "(555) 987-6543"
    }}

    Email content: "Sincerely, Emily Rogers Marketing Specialist emily.rogers@marketingco.com MarketingCo"
    Extracted signature information: {{
      "name": "Emily Rogers",
      "email": "emily.rogers@marketingco.com",
      "job_title": "Marketing Specialist",
      "company": "MarketingCo"
    }}

    Email content:{email_content}

    Extracted signature information:
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["email_content"]
    )
    return prompt

```
# Second Prompt Evaluation

## Mistral-Large-Latest Model Results (After Iteration):

* Count: 78
* Mean: 0.990927
* Standard Deviation: 0.018777
* Minimum: 0.927664
* 25th Percentile: 1.000000
* Median (50th Percentile): 1.000000
* 75th Percentile: 1.000000
* Maximum: 1.000000

## Open-Mistral-Nemo Model Results (After Iteration):

* Count: 78
* Mean: 0.882481
* Standard Deviation: 0.229348
* Minimum: 0.257969
* 25th Percentile: 0.948787
* Median (50th Percentile): 1.000000
* 75th Percentile: 1.000000
* Maximum: 1.000000

# Conclusion
The second prompt iteration showed a significant improvement in performance, especially for the Mistral-Large-Latest model. The mean cosine similarity increased, and the standard deviation decreased, indicating more consistent and accurate extractions. The use of examples in the prompt helped the models better understand the expected format and content, leading to fewer errors and more accurate extractions.

Overall, the iterative approach to prompt engineering, with careful evaluation and adjustment, led to a more robust and effective solution for extracting email signature information. The final results demonstrated the effectiveness of the refined prompt in handling a wide range of email signature scenarios.
