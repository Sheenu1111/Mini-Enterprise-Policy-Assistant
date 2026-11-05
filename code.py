
import os
import re
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POLICY_FILE = os.path.join(BASE_DIR, "policies.txt")

MODEL_NAME = "all-MiniLM-L6-v2"
POLICY_THRESHOLD = 0.40
SENTENCE_THRESHOLD = 0.25
MAX_SENTENCES = 2
LLM_MODEL = "claude-haiku-5-5"

DEMO_QUESTIONS = [
    "How many days can I work from home?",
    "Can I claim expenses from six months ago?",
    "What is the company's maternity leave policy?",
]

STOPWORDS = {
    "what", "whats", "is", "are", "the", "a", "an", "of", "for", "to",
    "do", "does", "can", "i", "we", "my", "our", "me", "how", "many",
    "much", "when", "where", "who", "which", "about", "company",
    "company's", "there", "any", "tell", "explain", "have", "get",
    "in", "on", "and", "or", "policy", "policies"
}

START_RE = re.compile(
    r"^\s*(\d+)[.)]?\s+(.+?)\s*(?:—|–|\s-\s|:)\s*(.*)$"
)


def load_policies(path):
    policies = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            match = START_RE.match(line)

            if match:
                num, name, text = match.groups()
                policies.append({
                    "num": int(num),
                    "name": name.strip(),
                    "text": text.strip()
                })

            elif policies:
                policies[-1]["text"] += " " + line

    return policies


def split_sentences(text):
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]


def topic_of(question):
    words = re.findall(r"[a-z']+", question.lower())
    return " ".join(
        word for word in words if word not in STOPWORDS
    ) or "this"


def llm_answer(question, policy):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None

    try:
        import anthropic

        reply = anthropic.Anthropic().messages.create(
            model=LLM_MODEL,
            max_tokens=200,
            system=(
                "Answer the employee's question using ONLY the policy text "
                "given. Reply in 1-2 short sentences. If the policy does not "
                "address the question, reply with exactly: NOT_FOUND"
            ),
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Policy ({policy['name']}): {policy['text']}\n\n"
                        f"Question: {question}"
                    )
                }
            ],
        )

        return reply.content[0].text.strip()

    except Exception:
        return None


class PolicyAssistant:

    def __init__(self, path=POLICY_FILE):
        self.policies = load_policies(path)

        if not self.policies:
            raise ValueError("No policies found in policies.txt")

        self.model = SentenceTransformer(MODEL_NAME)

        docs = [
            f"{policy['name']}. {policy['text']}"
            for policy in self.policies
        ]

        self.policy_vecs = self.model.encode(
            docs,
            normalize_embeddings=True
        )

    def retrieve(self, question):
        q_vec = self.model.encode(
            [question],
            normalize_embeddings=True
        )

        scores = cosine_similarity(q_vec, self.policy_vecs)[0]
        best = int(np.argmax(scores))

        return self.policies[best], float(scores[best]), q_vec

    def extract_answer(self, policy, q_vec):
        sentences = split_sentences(policy["text"])

        if not sentences:
            return "No relevant sentence found in this policy."

        sentence_vecs = self.model.encode(
            sentences,
            normalize_embeddings=True
        )

        scores = cosine_similarity(q_vec, sentence_vecs)[0]

        top = [
            i for i in np.argsort(scores)[::-1][:MAX_SENTENCES]
            if scores[i] >= SENTENCE_THRESHOLD
        ]

        if not top:
            return "No sufficiently relevant sentence found."

        return " ".join(sentences[i] for i in sorted(top))

    def ask(self, question):
        not_found = {
            "answer": (
                f"I could not find information about "
                f"{topic_of(question)} in the provided policies."
            ),
            "source": "None"
        }

        policy, score, q_vec = self.retrieve(question)

        if score < POLICY_THRESHOLD:
            return not_found

        answer = llm_answer(question, policy)

        if answer == "NOT_FOUND":
            return not_found

        if answer is None:
            answer = self.extract_answer(policy, q_vec)

        return {
            "answer": answer,
            "source": f"Policy {policy['num']} — {policy['name']}"
        }


def show(question, result):
    print(f"\nQ: {question}\n")
    print(f"A: {result['answer']}\n")
    print(f"Source: {result['source']}\n")


if __name__ == "__main__":

    bot = PolicyAssistant()
    args = sys.argv[1:]

    if args == ["--demo"]:
        for question in DEMO_QUESTIONS:
            show(question, bot.ask(question))

    elif args:
        question = " ".join(args)
        show(question, bot.ask(question))

    else:
        print("Ask a policy question (type 'quit' to exit).")

        while True:
            question = input("Question: ").strip()

            if question.lower() in {"quit", "exit", "q"}:
                break

            if question:
                show(question, bot.ask(question))
