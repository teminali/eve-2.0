import os
import PyPDF2
# import docx
from transformers import pipeline


class QuestionAnswering:
    def __init__(self, model_dir='models/q2a', knowledge_base=None, uid=None):
        self.model_dir = model_dir
        self.knowledge_base = knowledge_base
        self.uid = uid
        self.nlp = pipeline('question-answering', model=self.model_dir, tokenizer=self.model_dir)

    @staticmethod
    def extract_text_from_file(file_path):
        """Extracts text from a file, regardless of its format."""
        try:
            if file_path.endswith('.txt'):
                chat_logs_path = file_path + "/chat_ls/"
                with open(chat_logs_path, 'r') as file:
                    return file.read()
            elif file_path.endswith('.txt'):
                with open(file_path, 'r') as file:
                    return file.read()
            elif file_path.endswith('.pdf'):
                pdf_file = PyPDF2.PdfFileReader(file_path)
                text = ''
                for page in pdf_file.pages:
                    text += page.extract_text()
                return text
            # elif file_path.endswith('.docx'):
            #     doc = docx.Document(file_path)
            #     text = ''
            #     for paragraph in doc.paragraphs:
            #         text += paragraph.text
            #     return text
            else:
                raise Exception(f'Unsupported file format: {file_path}')
        except Exception as e:
            print(f"Error processing file '{file_path}': {e}")
            return ''

    def prepare_data_for_bert(self, question, directory_path):
        """Combines the text from all files in a directory and returns
        a dictionary suitable for use with the Bert QA model."""
        if not os.path.isdir(directory_path):
            raise Exception(f'Error: "{directory_path}" is not a valid directory')

        context = ''
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            context += self.extract_text_from_file(file_path) + '\n'
        return {
            'question': question,
            'context': context
        }

    def answer_question(self, question):
        """Ask a question and return the answer and confidence."""
        try:
            inputs = self.prepare_data_for_bert(question, self.knowledge_base)
            answer = self.nlp(inputs)
            confidence = answer['score']
            answer = answer['answer']
            response = {
                'answer': answer,
                'confidence': confidence
            }
            return response
        except Exception as e:
            print(f'Error answering question: {e}')
            response = {
                'answer': "",
                'confidence': 0
            }
            return response

# example usage
# if __name__ == '__main__':
#     q2a = QuestionAnswering(knowledge_base='knowledge_base/loan_application')
#     response = q2a.answer_question('When does the Loan Application window opens for 2022/2023?')
#     print(response)
