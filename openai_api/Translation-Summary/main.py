"""
This script will prompt user input for a country, then scrape the headlines of the selected newspaper website.
Once we grab the headlines from the newspaper, we send them to OpenAI for translation and summarization
"""
import requests
import bs4
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

### "Country" : (URL,HTML_TAG)
country_newspapers = {"Spain":('https://elpais.com/','.c_t'), 
                       "France":("https://www.lemonde.fr/",'.article__title-label')
                     }

def create_prompt():
    ### Get Country from user input, currently either Spain or France
    country = input("What country would you like a news summary for? ")
  
    ### Get country's URL newspaper and the HTML Tag for titles
    try:
        url,tag = country_newspapers[country]
    except:
        print("Sorry that country is not supported!")
        return
    
    ### Scrape the newspaper website
    results = requests.get(url)
    soup = bs4.BeautifulSoup(results.text,"lxml")
    
    ### Grab all the text of the top 3 headlines, each separated by a new line
    country_headlines = ''
    for item in soup.select(tag)[:3]:
        country_headlines += item.getText()+'\n'
      
    ### Create a prompt for OpenAI to translate the headlines to English
    prompt = "Detect the language of the news headlines below, then translate a summary of the headlines to English in a conversational tone:\n"
    return prompt + country_headlines

### Return the detected language from the headlines and their English translation
def translate_and_summary(prompt):
    response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.1, # Helps conversational tone a bit, optional
            max_tokens=200,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
            )
    print(response['choices'][0]['text'])

if __name__ == "__main__":
    prompt = create_prompt()
    translate_and_summary(prompt)
