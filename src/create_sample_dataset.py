import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def create_sample_dataset(output_path="data/sample_dataset.csv"):
    """
    Create a sample dataset with Hindi-English pairs including idioms and phrases.
    
    Args:
        output_path (str): Path to save the dataset
    """
    logger.info("[testing] Creating sample dataset with idioms and phrases")
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Sample data with regular sentences, idioms, and phrases
    data = {
        "english": [
            # Regular sentences
            "How are you doing today?",
            "I am going to the market.",
            "She is reading a book.",
            "They are playing cricket.",
            "The weather is very pleasant today.",
            
            # Idioms and phrases
            "It's raining cats and dogs.",
            "A piece of cake",
            "Break a leg",
            "The ball is in your court",
            "Bite the bullet",
            "The best of both worlds",
            "Speak of the devil",
            "Once in a blue moon",
            "Cost an arm and a leg",
            "Cut corners",
            
            # Context-dependent expressions
            "I'm feeling blue today.",
            "She has a green thumb.",
            "He kicked the bucket last year.",
            "Let's call it a day.",
            "We need to think outside the box."
        ],
        "hindi": [
            # Regular sentences
            "आज आप कैसे हैं?",
            "मैं बाजार जा रहा हूँ।",
            "वह किताब पढ़ रही है।",
            "वे क्रिकेट खेल रहे हैं।",
            "आज मौसम बहुत सुहावना है।",
            
            # Idioms and phrases corresponding to English ones
            "मूसलाधार बारिश हो रही है।",
            "बाएं हाथ का खेल",
            "शुभकामनाएं",
            "अब आपकी बारी है",
            "हिम्मत करके आगे बढ़ना",
            "दोनों जहान की खुशियां",
            "बात करते ही आ गया",
            "कभी-कभार",
            "बहुत महंगा पड़ना",
            "लापरवाही करना",
            
            # Context-dependent expressions
            "मैं आज उदास हूँ।",
            "उसे पौधों को उगाने का शौक है।",
            "वह पिछले साल चल बसा।",
            "आज के लिए बस करते हैं।",
            "हमें कुछ नया सोचना होगा।"
        ]
    }
    
    # Complex context-dependent examples
    english_complex = [
        "She spilled the beans about the surprise party.",
        "It's time to face the music after what you did.",
        "Stop beating around the bush and tell me the truth.",
        "That car must have cost him an arm and a leg.",
        "Finding a good job in this economy is like looking for a needle in a haystack.",
        "Don't put all your eggs in one basket with this investment.",
        "We're back to square one after that failed attempt.",
        "When pigs fly, that's when he'll apologize.",
        "It's not rocket science, anyone can learn this.",
        "The new manager is all bark and no bite."
    ]
    
    hindi_complex = [
        "उसने सरप्राइज़ पार्टी का भेद खोल दिया।",
        "अब तुम्हें अपने कर्मों का फल भुगतना होगा।",
        "बिना घुमा-फिरा कर मुझे सच बताओ।",
        "वह कार उसे बहुत महंगी पड़ी होगी।",
        "इस अर्थव्यवस्था में अच्छी नौकरी ढूंढना सुई के ढेर में सुई ढूंढने जैसा है।",
        "इस निवेश में सभी अंडे एक ही टोकरी में मत रखो।",
        "उस असफल प्रयास के बाद हम फिर से शुरुआत पर आ गए हैं।",
        "जब मुर्गे के दूध आएंगे, तभी वह माफी मांगेगा।",
        "यह कोई रॉकेट साइंस नहीं है, कोई भी इसे सीख सकता है।",
        "नया प्रबंधक सिर्फ बड़ी-बड़ी बातें करता है, करता कुछ नहीं।"
    ]
    
    # Add complex examples to the dataset
    data["english"].extend(english_complex)
    data["hindi"].extend(hindi_complex)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    logger.info(f"Created sample dataset with {len(df)} examples")
    logger.info(f"Dataset saved to {output_path}")
    
    return df

if __name__ == "__main__":
    create_sample_dataset() 