# Generative Projects with OpenAI APIs

Compilation of simple generative NLP and image projects using OpenAI's GPT-3.5 and DALL-E APIs. 

OpenAI Completion Parameters:

    - Model:
        - Always prefer to use the latest OpenAI model, but also consider the cost of API calls per 1000 tokens.
    - Prompt: 
        - The user prompt that tells the model what to return. Take advantage of the [OpenAI playground](https://platform.openai.com/playground) to understand how changing the prompt affects the model output.
    - Temperature (0 to 1): 
        - Lean towards 0 for simple tasks with a clear correct answer. 
        - Higher values means the model gets more creative and risky with its response. 
    - Max Tokens: 
        - Maximum number of tokens to generate the completion. 
        - Most models have a context length of 2048 tokens, but newer models support 4096 tokens.
    - Top P:
        - Alternative sampling to temperature (nucleus sampling), where the model considers the results of the tokens with Top P probability mass (0.1 = top 10% probability mass). 
        - Alter either Top P or Temperature, but **not** both at the same time.
    - N:
        - How many completions to generate for each prompt ie. running the same prompt multiple time (note: will cost money and consume token quota).  
        - Typically choose 1 to indicate only 1 completion.
    - Frequency Penalty (-2 to 2):
        - Positive values penalize new tokens based on their existing frequency in the text ie. reduce the model's likelihood to repeat the same line over and over again.
        - Negative values can be used to increase the model's likelihood of repetition in its response.
    - Presence Penalty (-2 to 2):
        - Positive values penalize new tokens based on whether they appear in the text so far ie. increase the model's likelihood to talk about new topics.
        - Negative values can be used to increase the model's likelihood of repetition in its response.
    - For complete details, refer to the [OpenAI API guide](https://platform.openai.com/docs/api-reference).


DALLE-2 Parameters:

    - Image API: 
        - Takes in a text prompt and returns an image that matches what is in the text prompt.
    - Image size:
        - Choose among 256x256, 512x512, or 1024x1024 (pay attention to pricing).
    - N:
        - Request between 1 to 10 images via the image API

**Code Status**: Last updated in September 2023. OpenAI's service availability is limited in HK, and the codes in this repository will not be actively maintained inline with the latest models. I might expand the repository later with other relevant generative AI tools eg. ChatGPT-4, Langchain, Midjourney.

Credit to Pierian Training for the hands-on training with OpenAI API services and prompt engineering.
