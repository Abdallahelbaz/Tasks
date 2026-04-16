from langclassifier import LangClassifier
from arabictext import ArabicText
from englishtext import EnglishText


def main():

    text= input("Enter Your Text:\n")
    print(f"your Text is: {text}\n")

    lc=LangClassifier()
    ac=ArabicText()
    ec=EnglishText()

    lang_result=lc.predict_language(text)

    if lang_result==0:
        result= ac.classify_text(text)
        print(f"The Language is Arabic. \nThe Classification of the text is: {result}")
    else:
        result= ec.classify_text(text)
        print(f"The Language is English. \nThe Classification of the text is: {result}")


if __name__ == "__main__":
    while(1):
        main()