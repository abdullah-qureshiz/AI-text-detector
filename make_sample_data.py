"""make_sample_data.py - creates a tiny DEMO dataset (made-up sentences)."""
import csv
import os
import random

random.seed(7)

ai_sentences = [
    "Artificial intelligence has transformed the way organizations operate in the modern world.",
    "It is important to note that effective communication is essential for long term success.",
    "Furthermore, technology plays a crucial role in improving efficiency and productivity.",
    "In conclusion, a balanced approach offers numerous benefits for individuals and society.",
    "Moreover, education provides valuable opportunities for personal and professional growth.",
    "Overall, the integration of digital tools enhances collaboration and streamlines workflows.",
    "Additionally, sustainable practices contribute significantly to the well being of future generations.",
    "This comprehensive approach ensures that stakeholders can achieve optimal outcomes.",
    "There are several key factors that contribute to the overall success of any project.",
    "Understanding these concepts is vital for navigating the complexities of the digital landscape.",
    "By leveraging innovative solutions, businesses can foster growth and maintain a competitive advantage.",
    "Ultimately, continuous learning empowers individuals to adapt to a rapidly changing environment.",
    "In today's fast paced world, time management is a fundamental skill for achieving goals.",
    "Healthy habits play an essential role in enhancing both physical and mental well being.",
    "Therefore, it is crucial to consider multiple perspectives when evaluating complex issues.",
]

human_sentences = [
    "Honestly I forgot my umbrella again and got soaked on the way to work.",
    "We tried that cafe near the station last week. Coffee was fine, the cake was amazing though!",
    "My brother keeps borrowing my charger and never returns it, which drives me nuts.",
    "I don't know, maybe I'll just stay home tonight and watch something dumb.",
    "The bus was late. Again. Third time this week, so I walked.",
    "Grandma's kitchen always smelled like cardamom and burnt toast, weirdly comforting.",
    "Okay so the exam wasn't as bad as I feared, but question five? Total nightmare.",
    "He laughed so hard he nearly dropped the tray, and everybody stared at us.",
    "I wanted to fix the bike myself, ended up with grease all over my shirt and a bent wheel.",
    "Not gonna lie, that match yesterday had me shouting at the TV like a madman.",
    "We argued about pizza toppings for twenty minutes. Pineapple lost, obviously.",
    "The street was quiet except for a dog barking somewhere and the hum of a fridge next door.",
    "She texted me at 2 am about a spider. A spider! I live forty minutes away.",
    "My phone died halfway through the map, so we just guessed and somehow it worked out.",
    "Rain, mud, cold tea, a broken tent. Best camping trip ever, I swear.",
]

rows = []
for _ in range(150):
    rows.append((" ".join(random.sample(ai_sentences, random.randint(4, 7))), 1))      # 1 = AI
    rows.append((" ".join(random.sample(human_sentences, random.randint(4, 7))), 0))   # 0 = human
random.shuffle(rows)

os.makedirs("data", exist_ok=True)
with open("data/sample_data.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "label"])
    writer.writerows(rows)

print(f"Created data/sample_data.csv with {len(rows)} rows")