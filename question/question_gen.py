from transformers import T5ForConditionalGeneration,T5Tokenizer
from transformers import AutoConfig
trained_model_path = '/mnt/c/Users/sanatan/OneDrive/Desktop/Question-MCQ-_Answer_Generation-main/UI/MCQ/question/model/t5/model'
trained_tokenizer = '/mnt/c/Users/sanatan/OneDrive/Desktop/Question-MCQ-_Answer_Generation-main/UI/MCQ/question/model/t5/tokenizer'

model = T5ForConditionalGeneration.from_pretrained(trained_model_path,local_files_only=False)
tokenizer = T5Tokenizer.from_pretrained(trained_tokenizer,local_files_only=False)



def question_g(summarized_text,keywords):
    questions = []
    for i in keywords:
        text = "context: {} answer: {}</s>".format(summarized_text,i)
        encoding = tokenizer.encode_plus(text,max_length =512, padding=True, return_tensors="pt")
        input_ids,attention_mask  = encoding["input_ids"].to('cpu'), encoding["attention_mask"].to('cpu')

        model.eval()
        beam_outputs = model.generate(
            input_ids=input_ids,attention_mask=attention_mask,
            max_length=72,
            early_stopping=True,
            num_beams=5,
            num_return_sequences=1

        )

        for beam_output in beam_outputs:
            sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
        questions.append(sent)
    return(questions)

