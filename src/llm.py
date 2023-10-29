import openai
from src.common.path_setup import output_dir
import os 
from io import StringIO
import pandas as pd
import time
from src.common.logger import logger as sc_logger


class AZURE_OPENAI_MODEL:
    def __init__(self, model,resource, key,api_type = 'azure', top_p = 1,temperature=0.8,max_tokens=2000):
        self.model = model 
        self.resource = resource
        self.key = key
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.api_type = api_type

    def query_llm(self, df=None):
        openai.api_type = "azure"
        openai.api_base = f"https://{self.resource}.openai.azure.com/"
        openai.api_version = "2023-03-15-preview"
        openai.api_key = self.key

        messages = self.prepare_messages(df)

        response = openai.ChatCompletion.create(
            engine=self.model,
            messages=messages,
            temperature=float(self.temperature),
            max_tokens=int(self.max_tokens),
            top_p=float(self.top_p),
            stream=False,
        )

        answer = response.choices[0].message.content
        try:
            # Extract the answer from the response. Answer is enclosed inside ``` ```
            parsed_answer = answer.split("```")[1]
            # print(parsed_answer[:100])
            data_io = StringIO(parsed_answer.strip())

            results_df = pd.read_csv(data_io,na_values='null')
        except Exception as e:
            sc_logger.error(f"Error in parsing the answer>>> {str(e)}")
            return None

        sc_logger.info(f"input_shape >> {df.shape}")
        sc_logger.info(f"output_shape >> {results_df.shape}")


        return results_df
    

    def prepare_messages(self, df=None):

        SYSTEM_MESSAGE = """
            You are chatbot that is used for gender classification based on names of people living in Karnataka Region India.
            Rules: 
            1. Gender classification would be either MALE/FEMALE.
            2. Output should be in csv format strict. 
            3. Output should be enclosed inside ```
            4. Output should have two columns names and predicted_gender.
            5. Remove new line characters in the beginning of the output and end.
            
            Example Output:
            names,predicted_gender
            Ramesh Kumar,MALE
            Sita sharma,FEMALE"""
        messages = [{"role": "system", "content": SYSTEM_MESSAGE}]

        msg = {}
        msg["role"] = "user"
        msg["content"] = f""" Here are list of names: {df['name'].tolist()}"""
        messages.append(msg)

        return messages
    
    def parse_using_llm(self, raw_texts=""):
        openai.api_type = "azure"
        openai.api_base = f"https://{self.resource}.openai.azure.com/"
        openai.api_version = "2023-03-15-preview"
        openai.api_key = self.key

        messages = self.prepare_messages_for_parsing(raw_texts)
        # print("messages>>", messages)

        response = openai.ChatCompletion.create(
            engine=self.model,
            messages=messages,
            temperature=float(self.temperature),
            max_tokens=int(self.max_tokens),
            top_p=float(self.top_p),
            stream=False,
        )

        answer = response.choices[0].message.content
        try:
            # Extract the answer from the response. Answer is enclosed inside ``` ```
            if "```" in answer:
                parsed_answer = answer.split("```")[1]
            else:
                parsed_answer = answer
            # print(parsed_answer)
            data_io = StringIO(parsed_answer.strip())

            results_df = pd.read_csv(data_io,delimiter=';', na_values='null')
        except Exception as e:
            sc_logger.error(f"Error in parsing the answer>>> {str(e)}")
            return None

        sc_logger.info(f"output_shape >> {results_df.shape}")


        return results_df
    

    def prepare_messages_for_parsing(self, raw_texts):

        SYSTEM_MESSAGE = """
            You are chatbot that is used for parsing voters info extracted using ocr.
            Rules: 
            1. Output should be in csv format strict with delimiter ;. 
            2. Output should be enclosed inside ```
            3. Remove new line characters in the beginning of the output and end.
            4. Input would be lists of texts. Each item of the list is specific to a particular voter.     
            5. Id can be numeric only. If it is other than numeric or it is not able to parse, keep it as null.
            Example Input:
            Here are list of voter related information:["20 TNH3552262\\nName: Jtender Kumar Singh K\\nFather\'s Name : Rajkumar Singh\\nHouse Number : 46,\\nAge: 66 Gender: MALE Photo is\\nAvailable", "L__ 8] TNH3421906\\n\\nName: Aslam Uddin\\n\\nFather\'s Name : Bodrul Hoque\\n\\nHouse Number: 18\\n\\nAge: 28 Gender: MALE Photo is\\nAvailable", "28 TNH3421898\\nName: Romij Uddin Choudhary\\nFather\'s Name : Junab Ali Choudary\\nHouse Number: 18\\nAge: 32 Gender: MALE Photo is\\nAvailable", "La TNH2625077\\n\\nName: Suma B\\n\\nFather\'s Name : Late Vishnu Bahaddur\\n\\nHouse Number: 17/B\\n\\nAge: 34 Gender: FEMALE Photo is\\nAvailable"]

            Example Output:            id;voter_id;name;house_number;age;gender;parent_or_spouse_name_only  
            20;TNH3552262;Jtender Kumar Singh K;46;66;MALE;Rajkumar Singh  
            8;TNH3421906;Aslam Uddin;18;28;MALE;Bodrul Hoque  
            28;TNH3421898;Romij Uddin Choudhary;18;32;MALE;Junab Ali Choudary  
            null;TNH2625077;Suma B;17/B;34;FEMALE;Late Vishnu Bahaddur"""
        messages = [{"role": "system", "content": SYSTEM_MESSAGE}]

        msg = {}
        msg["role"] = "user"
        msg["content"] = f"""Here are list of voter related information: {raw_texts}"""
        messages.append(msg)

        return messages
    
    def save_results(self, df,results_df,file_name):
        """
        This function saves the information to the csv file
        """

        # combine the dataframe with the results left join merge on names and name
        output_df = pd.merge(df, results_df, left_on="name", right_on="names", how="left")

        # drop the names column
        output_df.drop(columns=["names"], inplace=True)

        # Save the DataFrame to a CSV file
        csv_file_path = os.path.join(output_dir, f"{file_name}_llm_output.csv")
        output_df.to_csv(csv_file_path, sep=";", index=False)

        return csv_file_path
    

from dotenv import load_dotenv
import os
from src.common.path_setup import output_dir
import pandas as pd
import numpy as np

load_dotenv()

def gender_classification_using_llm():

    model = os.environ.get('AZURE_OPENAI_MODEL')
    resource = os.environ.get('AZURE_OPENAI_RESOURCE')
    key = os.environ.get('AZURE_OPENAI_KEY')
    temperature = os.environ.get('AZURE_OPENAI_TEMPERATURE')
    top_p = os.environ.get('AZURE_OPENAI_TOP_P')
    max_tokens = os.environ.get('AZURE_OPENAI_MAX_TOKENS')

    llm = AZURE_OPENAI_MODEL(model,resource,key,temperature=temperature,top_p=top_p,max_tokens=max_tokens)
    
    file_path = os.path.join(output_dir, 'electoral_rolls_parsed_by_llm_output.csv')
    df = pd.read_csv(file_path, sep=';')

    processed_batches = []

    # split it into the batches of 100 
    for start in range(0, len(df), 100):  # Here, 100 is the size of your batch
        end = start + 100
        df_batch = df[start:end]
        
        # Process the batch and store the result
        result_batch = llm.query_llm(df_batch)
        processed_batches.append(result_batch)
        time.sleep(2)

    answers_df = pd.concat(processed_batches).reset_index(drop=True)

    csv_file_path = llm.save_results(df,answers_df,'voters_info.csv')

    # statistical analysis of the results
    # read the csv file
    llm_results_df = pd.read_csv(csv_file_path, sep=';')

    # compare gender and predicted_gender columns and on the basis of it calculate accuracy

    # create a new column called is_correct
    llm_results_df["is_correct"] = llm_results_df.apply(
        lambda x: x['gender'] == x['predicted_gender'], axis=1
    )
    sc_logger.info(f"llm_results>>{llm_results_df.head(100)}")
    llm_results_df.to_csv(os.path.join(output_dir, 'llm_results_analysis.csv'),sep=';', index=False)


if __name__ == '__main__':
    gender_classification_using_llm()

