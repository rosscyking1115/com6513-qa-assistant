import json
import os
import re
import fitz


PDF_FILES = {
    "computer_security": "data/raw/Computer_security.pdf",
    "stuxnet": "data/raw/Stuxnet.pdf",
    "wannacry": "data/raw/WannaCry_ransomware_attack.pdf"
}

QA_PAIRS = [
    ("q01", "computer_security", "What is computer security also known as?", "cybersecurity, digital security, or information technology (IT) security"),
    ("q02", "computer_security", "Where are most discovered vulnerabilities documented?", "Common Vulnerabilities and Exposures (CVE) database"),
    ("q03", "computer_security", "What is a backdoor in a computer system?", "any secret method of bypassing normal authentication or security controls"),
    ("q04", "computer_security", "What are Denial-of-service attacks designed to do?", "make a machine or network resource unavailable to its intended users"),
    ("q05", "computer_security", "What is eavesdropping?", "the act of surreptitiously listening to a private computer conversation (communication), usually between hosts on a network"),
    ("q06", "computer_security", "What is malware?", 'any software code or computer program "intentionally written to harm a computer system or its users."'),
    ("q07", "computer_security", "What is phishing?", "the attempt to acquire sensitive information such as usernames, passwords, and credit card details directly from users by deceiving the users"),
    ("q08", "computer_security", "How much had business email compromise scams cost US businesses by early 2016?", "more than $2 billion in about two years"),
    ("q09", "computer_security", "What percentage of cyber security incidents involved internal actors according to the Verizon Data Breach Investigations Report 2020?", "30%"),
    ("q10", "computer_security", "What is the CIA triad considered the foundation of?", "information security"),
    ("q11", "stuxnet", "When was Stuxnet first uncovered?", "17 June 2010"),
    ("q12", "stuxnet", "What type of systems does Stuxnet target?", "supervisory control and data acquisition (SCADA) systems"),
    ("q13", "stuxnet", "Who discovered Stuxnet?", "Sergey Ulasen from a Belarusian antivirus company VirusBlokAda"),
    ("q14", "stuxnet", "What did Stuxnet reportedly destroy?", "almost one-fifth of Iran's nuclear centrifuges"),
    ("q15", "stuxnet", "How is Stuxnet typically introduced to the target environment?", "via an infected USB flash drive"),
    ("q16", "stuxnet", "According to Symantec, what were the three main affected countries in the early days of the Stuxnet infection?", "Iran, Indonesia and India"),
    ("q17", "stuxnet", "How many zero-day attacks did Stuxnet use against Windows systems?", "four zero-day attacks"),
    ("q18", "stuxnet", "What is the file size of Stuxnet?", "half a megabyte"),
    ("q19", "stuxnet", "How long did Symantec estimate it would have taken to prepare Stuxnet?", "six months"),
    ("q20", "stuxnet", "By how much did the centrifuge operational capacity at Natanz drop?", "30 percent"),
    ("q21", "wannacry", "What time did the WannaCry attack begin?", "07:44 UTC on 12 May 2017"),
    ("q22", "wannacry", "Who developed the EternalBlue exploit used by WannaCry?", "the United States National Security Agency (NSA)"),
    ("q23", "wannacry", "Who discovered the WannaCry kill switch?", "Marcus Hutchins"),
    ("q24", "wannacry", "What programming tool was used to create WannaCry?", "Microsoft Visual C++ 6.0"),
    ("q25", "wannacry", "How much did WannaCry demand as ransom within three days?", "around US$300 in bitcoin"),
    ("q26", "wannacry", "How many computers were infected within a day of the WannaCry attack?", "more than 230,000 computers in over 150 countries"),
    ("q27", "wannacry", "How many ransom payments were made during the WannaCry attack?", "327 payments totaling US$130,634.77"),
    ("q28", "wannacry", "According to Kaspersky Lab, what were the four most affected countries?", "Russia, Ukraine, India and Taiwan"),
    ("q29", "wannacry", "How many NHS devices may have been affected by WannaCry?", "up to 70,000 devices"),
    ("q30", "wannacry", "What was the estimated cost of the WannaCry attack to the NHS?", "£92 million in disruption to services and IT upgrades")
]


def clean_text(text):
    """Clean extracted PDF text."""
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()



def extract_pdf_text(pdf_path):
    """Extract the main text from a PDF."""
    doc = fitz.open(pdf_path)
    pages_text = []

    for page in doc:
        text = page.get_text()

        stop_markers = ["\nReferences\n", "\nSee also\n", "\nExternal links\n"]
        for marker in stop_markers:
            index = text.lower().find(marker.lower())
            if index != -1:
                text = text[:index]
                pages_text.append(text)
                doc.close()
                return clean_text("\n".join(pages_text))

        pages_text.append(text)

    doc.close()
    return clean_text("\n".join(pages_text))



def main():
    os.makedirs("data", exist_ok=True)

    print("Extracting text from PDFs...")
    documents = {}

    for key in PDF_FILES:
        pdf_path = PDF_FILES[key]

        if not os.path.exists(pdf_path):
            print("WARNING: File not found:", pdf_path)
            documents[key] = ""
        else:
            text = extract_pdf_text(pdf_path)
            documents[key] = text
            print(key, ":", len(text.split()), "words")

    print()
    print("Building input.json...")

    dataset = []
    for item in QA_PAIRS:
        qid = item[0]
        doc_key = item[1]
        question = item[2]

        dataset.append({
            "question_id": qid,
            "question": question,
            "document": documents[doc_key]
        })

    with open("data/input.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print("Saved", len(dataset), "QA pairs to data/input.json")

    gold_answers = []
    for item in QA_PAIRS:
        gold_answers.append({
            "question_id": item[0],
            "answer": item[3]
        })

    with open("data/gold_answers.json", "w", encoding="utf-8") as f:
        json.dump(gold_answers, f, indent=2, ensure_ascii=False)

    print("Saved gold answers to data/gold_answers.json")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
