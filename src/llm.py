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

            results_df = pd.read_csv(data_io)
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
        output_df.to_csv(csv_file_path, index=False)

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
    
    file_path = os.path.join(output_dir, 'voters_info.csv')
    df = pd.read_csv(file_path)

    processed_batches = []

    # split it into the batches of 100 
    for start in range(0, len(df), 50):  # Here, 100 is the size of your batch
        end = start + 50
        df_batch = df[start:end]
        
        # Process the batch and store the result
        result_batch = llm.query_llm(df_batch)
        processed_batches.append(result_batch)
        time.sleep(10)

    answers_df = pd.concat(processed_batches).reset_index(drop=True)

    csv_file_path = llm.save_results(df,answers_df,'voters_info')

    # statistical analysis of the results
    # read the csv file
    llm_results_df = pd.read_csv(csv_file_path)

    # compare gender and predicted_gender columns and on the basis of it calculate accuracy

    # create a new column called is_correct
    llm_results_df["is_correct"] = llm_results_df.apply(
        lambda x: x['gender'] == x['predicted_gender'], axis=1
    )
    sc_logger.info(f"llm_results>>{llm_results_df.head(100)}")
    llm_results_df.to_csv(os.path.join(output_dir, 'llm_results_analysis.csv'), index=False)


if __name__ == '__main__':
    gender_classification_using_llm()

