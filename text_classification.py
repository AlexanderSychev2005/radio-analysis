from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import json
from decouple import config

GOOGLE_API_KEY = config("GOOGLE_API_KEY")


class RadioClassification(BaseModel):
    category: str = Field(
        description="Клас радіоперехоплення. Обери один з: 'Розвідка', 'Медична евакуація', 'Артилерія', 'Логістика', 'Невідомо'"
    )
    confidence_score: float = Field(
        description="Впевненість моделі у класифікації від 0.0 до 1.0"
    )
    summary: str = Field(description="Короткий зміст повідомлення одним реченням")
    entities: list[str] = Field(
        description="Список ключових сутностей: позивні, координати, кількість техніки або людей"
    )


def setup_classifier():
    parser = JsonOutputParser(pydantic_object=RadioClassification)

    template = """
    Ти — досвідчений військовий аналітик-зв'язківець. 
    Твоє завдання — проаналізувати та класифікувати текст радіоперехоплення.
    Враховуй, що текст отримано через систему розпізнавання мовлення (ASR), 
    тому в ньому можуть бути пропущені розділові знаки або спотворені слова через радіоперешкоди.
    
    Текст перехоплення: "{text}"
    
    {format_instructions}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.1)

    chain = prompt | llm | parser
    return chain


if __name__ == "__main__":
    transcribed_text = "база я сокіл квадрат шість чотири бачу рух броні дві одиниці йдуть на південь прийом"

    classifier_chain = setup_classifier()

    print("Text analysis by Gemini..\n")
    try:
        result = classifier_chain.invoke({"text": transcribed_text})
        print(json.dumps(result, indent=4, ensure_ascii=False))
    except Exception as e:
        print(f"Classification Error: {e}")
