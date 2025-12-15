from flask import Flask, render_template, request
import pickle
import random

app = Flask(__name__, template_folder='.', static_folder='.', static_url_path='')


try:
    with open('spam_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except:
    print("Error: Run train_model.py first!")
    exit()


def get_psychological_trigger(text):
    text = text.lower()
    
   
    if any(word in text for word in ['police', 'arrest', 'banned', 'hacked', 'court', 'jail']):
        return "FEAR (Scammer is trying to scare you)"
    

    if any(word in text for word in ['lottery', 'winner', 'cash', 'prize', 'dollars', 'million']):
        return "GREED (Scammer is using money to trick you)"
    

    if any(word in text for word in ['urgent', 'immediately', 'now', 'expires', '24 hours']):
        return "URGENCY (Scammer wants you to panic)"
    

    return "General Spam"


funny_replies = [
    "Oh wow! Tell me more about this amazing price.",
    "I will send the money, but first solve this math...",
    "Sorry, my goldfish ate my credit card.",
    "Can I pay you in face to face cash money?",
    "Please contact with your father."
]









@app.route('/')
def home():

    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
       
        email_content = request.form['email_content']
        
        data_vectorized = vectorizer.transform([email_content])
        
        prediction = model.predict(data_vectorized)
        
        probability = model.predict_proba(data_vectorized)[0][1] * 100
        
        
        result = ""
        trigger_warning = ""
        auto_reply = ""
        
        
        if prediction[0] == 1:
            result = "SPAM Detected! "
            trigger_warning = get_psychological_trigger(email_content)
            auto_reply = random.choice(funny_replies)
            
        else:
            result = "Not Spam (Safe) "
            trigger_warning = "Nothing (Content looks safe)"
            auto_reply = "Replay in your own as per context."


        return render_template('index.html', 
                               prediction_text=result, 
                               email_content=email_content,
                               probability=round(probability, 2),
                               trigger=trigger_warning,
                               reply=auto_reply)

if __name__ == '__main__':
    app.run(debug=True)