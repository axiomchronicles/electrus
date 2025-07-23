
import requests

OPENROUTER_API_KEY = "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"

def chatapi():
    user_message = """You are a creative AI assistant that outputs valid JSON only.

You are generating content for a meme-powered parody horror game called “Brainrot Trials – Italian Food Horror Edition”. In this game, players must guess which surreal, AI-generated Italian food-themed brainrot character is being described.

These characters are inspired by viral internet memes and videos, and have absurd names and personalities such as:

- Tung Tung Tung Sahur – 🥁 A wooden protector who drums before dawn.
- Bombardino Crocodilo – 🐊 A warlike crocodile who drops bombs while singing lullabies.
- Ballerina Cappuccina – ☕ A coffee-headed ballerina who pirouettes through shadows.
- Trallalero Trallala – 👟 A sneaker-wearing shark who sings in opera.
- Brr Brr Patapim – 🎩 A baboon-tree hybrid that casts spells with its top hat.
- Chimpanzini Bananini – 🍌 A genetically modified chimpanzee who speaks only in rhymes.

Your task is to generate a **single game round** in the following JSON structure:

```json
{
  "gameRound": {
    "id": integer,                           // A unique round number
    "clue": string,                          // A poetic, creepy or funny clue describing the character
    "options": [string, string, string],     // Three possible character names (must include the real one)
    "correct": integer,                      // Index (0–2) of the correct name in the options
    "character": string                      // An emoji representing the character’s nature
  },
  "winningMessage": string,                  // A short dramatic line for when the player guesses correctly
  "scoreResponse": {
    "bigLine": string,                       // A bold, themed score summary
    "phrases": [string, string, string]      // Three creepy or absurd scoring comments
  }
}
Respond **with exactly that JSON object**, with no extra text, comments, or formatting.



    """

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://yourdomain.com",  
            "X-Title": "MyCoolApp", 
        },
        json={
            "model": "moonshotai/kimi-k2:free",  # Free model
            "messages": [
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        }
    )

    if response.status_code == 200:
        ai_reply = response.json()["choices"][0]["message"]["content"]
        return print({"reply": ai_reply})
    else:
        return print({
            "error": f"Failed with status {response.status_code}",
            "details": response.text
        })
    
if __name__ == "__main__":
    chatapi()