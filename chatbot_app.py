import nltk
import random
import os
import ssl
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
   
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath('nltk_data'))
nltk.download('punkt')

tags = []
patterns = []
responses = {}
intents = [
    {
        'tag':'greeting',
        'patterns':['Hi', 'Hello', 'Hey', 'Whats up', 'How are you'],
        'responses':['Hi there', 'Hello', 'Hey', 'Nothing much', 'Im fine, thank you']                                       
    },
    {
        'tag':'name',
        'patterns':['What is your name', 'Do you have a name', 'Who are you'],
        'responses':['I am a chatbot', 'I am a virtual assistant', 'You can call me Chatbot']         
    },
    {
        'tag':'goodbye',
        'patterns':['Bye', 'See you later', 'Goodbye', 'Take care'],
        'responses':['Goodbye', 'See you later', 'Take care']
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses": ["You're welcome", "No problem", "Glad I could help"]
    },
    {
        "tag": "about",
        "patterns": ["What can you do", "Who are you", "What are you", "What is your purpose"],
        "responses": ["I am a chatbot", "My purpose is to assist you", "I can answer questions and provide assistance"]
    },
    {
        "tag": "help",
        "patterns": ["Help", "I need help", "Can you help me", "What should I do"],
        "responses": ["Sure, what do you need help with?", "I'm here to help. What's the problem?", "How can I assist you?"]
    },
    {
        "tag": "age",
        "patterns": ["How old are you", "What's your age"],
        "responses": ["I don't have an age. I'm a chatbot.", "I was just born in the digital world.", "Age is just a number for me."]
    },
    {
        "tag": "weather",
        "patterns": ["What's the weather like", "How's the weather today"],
        "responses": ["I'm sorry, I cannot provide real-time weather information.", "You can check the weather on a weather app or website."]
    },
    {
        "tag": "budget",
        "patterns": ["How can I make a budget", "What's a good budgeting strategy", "How do I create a budget"],
        "responses": ["To make a budget, start by tracking your income and expenses. Then, allocate your income towards essential expenses like rent, food, and bills. Next, allocate some of your income towards savings and debt repayment. Finally, allocate the remainder of your income towards discretionary expenses like entertainment and hobbies.", "A good budgeting strategy is to use the 50/30/20 rule. This means allocating 50% of your income towards essential expenses, 30% towards discretionary expenses, and 20% towards savings and debt repayment.", "To create a budget, start by setting financial goals for yourself. Then, track your income and expenses for a few months to get a sense of where your money is going. Next, create a budget by allocating your income towards essential expenses, savings and debt repayment, and discretionary expenses."]
    },
    {
        "tag": "credit_score",
        "patterns": ["What is a credit score", "How do I check my credit score", "How can I improve my credit score"],
        "responses": ["A credit score is a number that represents your creditworthiness. It is based on your credit history and is used by lenders to determine whether or not to lend you money. The higher your credit score, the more likely you are to be approved for credit.", "You can check your credit score for free on several websites such as Credit Karma and Credit Sesame."]
    },
    {
        "tag": "investing",
        "patterns": ["How can I start investing", "What is the best way to invest money", "How do I invest in the stock market"],
        "responses": ["To start investing, you can open a brokerage account with an online broker and start investing in stocks, bonds, and other securities. You can also invest in mutual funds and exchange-traded funds (ETFs) to diversify your portfolio.", "The best way to invest money depends on your financial goals, risk tolerance, and investment timeline. It's a good idea to start by setting financial goals for yourself and then creating an investment plan to achieve those goals.", "To invest in the stock market, you can open a brokerage account with an online broker and start buying and selling stocks. You can also invest in mutual funds and exchange-traded funds (ETFs) to diversify your portfolio."]
    },
    {
        "tag": "insurance",
        "patterns": ["Why do I need insurance", "What are the different types of insurance", "How do I choose the right insurance"],
        "responses": ["Insurance is important because it helps protect you and your assets from financial loss. There are many different types of insurance, including health insurance, life insurance, auto insurance, and homeowners insurance. To choose the right insurance, consider your financial situation, risk tolerance, and insurance needs."]
    },
    {
        "tag": "loan",
        "patterns": ["How do I get a loan", "What are the different types of loans", "How do I choose the right loan"],
        "responses": ["To get a loan, you can apply for one at a bank, credit union, or online lender. You will need to provide information about your income, credit history, and financial situation. The lender will review your application and determine whether or not to approve your loan.", "There are many different types of loans, including personal loans, auto loans, student loans, and home loans. To choose the right loan, consider your financial situation, credit history, and loan needs."]
    },
    {
        "tag": "savings",
        "patterns": ["How can I save money", "What are some good saving tips", "How do I start saving"],
        "responses": ["To save money, start by tracking your income and expenses to get a sense of where your money is going. Then, create a budget and allocate a portion of your income towards savings. You can also automate your savings by setting up automatic transfers from your checking account to your savings account.", "Some good saving tips include setting financial goals for yourself, cutting back on discretionary expenses, and finding ways to increase your income. You can also save money by shopping for deals, using coupons, and taking advantage of discounts."]
    },
    {
        "tag": "taxes",
        "patterns": ["How do I file my taxes", "What are some tax deductions", "How can I reduce my tax bill"],
        "responses": ["To file your taxes, you can use tax software or hire a tax professional to help you. You will need to gather information about your income, expenses, and deductions and then file your tax return with the IRS.", "Some common tax deductions include the standard deduction, mortgage interest deduction, and charitable contribution deduction. You can also deduct medical expenses, state and local taxes, and student loan interest.", "To reduce your tax bill, consider contributing to a retirement account, taking advantage of tax credits, and itemizing your deductions. You can also reduce your taxable income by investing in tax-advantaged accounts and deferring income to future years."]
    },
    {
        "tag": "student_loans",
        "patterns": ["How do I pay off my student loans", "What are some student loan forgiveness programs", "How can I lower my student loan payments"],
        "responses": ["To pay off your student loans, consider making extra payments, refinancing your loans, and enrolling in an income-driven repayment plan. You can also take advantage of student loan forgiveness programs and employer-sponsored repayment assistance programs.", "Some student loan forgiveness programs include Public Service Loan Forgiveness (PSLF), Teacher Loan Forgiveness, and Perkins Loan Cancellation. You can also qualify for loan forgiveness through income-driven repayment plans and loan repayment assistance programs.", "To lower your student loan payments, consider enrolling in an income-driven repayment plan, refinancing your loans, or consolidating your loans. You can also take advantage of deferment and forbearance options to temporarily pause your payments."]
    },
    {
        "tag": "retirement",
        "patterns": ["How can I save for retirement", "What are some retirement planning tips", "When should I start saving for retirement"],
        "responses": ["To save for retirement, consider contributing to a retirement account such as a 401(k) or IRA. You can also invest in stocks, bonds, and other securities to grow your retirement savings. It's a good idea to start saving for retirement as early as possible to take advantage of compound interest and grow your savings over time.", "Some retirement planning tips include setting financial goals for yourself, creating a retirement budget, and estimating your retirement expenses. You can also work with a financial advisor to create a retirement plan and invest your savings wisely."]
    },
    {
        "tag": "investments",
        "patterns": ["What are some good investment options", "How can I invest my money wisely", "What is the best way to invest for the future"],
        "responses": ["Some good investment options include stocks, bonds, mutual funds, and exchange-traded funds (ETFs). You can also invest in real estate, commodities, and other alternative investments to diversify your portfolio and reduce risk.", "To invest your money wisely, consider your financial goals, risk tolerance, and investment timeline. It's a good idea to create an investment plan and diversify your portfolio to reduce risk. You can also work with a financial advisor to help you make informed investment decisions.", "The best way to invest for the future depends on your financial goals, risk tolerance, and investment timeline. It's a good idea to start by setting financial goals for yourself and then creating an investment plan to achieve those goals. You can also work with a financial advisor to help you make informed investment decisions."]
    },
    {
        "tag": "debt",
        "patterns": ["How can I pay off my debt", "What are some debt relief options", "How can I reduce my debt"],
        "responses": ["To pay off your debt, consider creating a debt repayment plan, making extra payments, and consolidating your debt. You can also work with a credit counselor or debt relief company to help you negotiate with creditors and reduce your debt.", "Some debt relief options include debt settlement, debt consolidation, and bankruptcy. You can also take advantage of debt management plans and debt relief programs to reduce your debt and improve your financial situation.", "To reduce your debt, consider creating a budget, cutting back on discretionary expenses, and finding ways to increase your income. You can also negotiate with creditors to lower your interest rates and monthly payments."]
    },
    {
        "tag": "mortgage",
        "patterns": ["How do I get a mortgage", "What are the different types of mortgages", "How can I qualify for a mortgage"],
        "responses": ["To get a mortgage, you can apply for one at a bank, credit union, or mortgage lender. You will need to provide information about your income, credit history, and financial situation. The lender will review your application and determine whether or not to approve your mortgage.", "There are many different types of mortgages, including fixed-rate mortgages, adjustable-rate mortgages, and government-insured mortgages. To choose the right mortgage, consider your financial situation, credit history, and mortgage needs.", "To qualify for a mortgage, you will need to meet the lender's requirements for income, credit score, and debt-to-income ratio. You will also need to provide documentation such as pay stubs, tax returns, and bank statements."]
    }
 ]
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)
    responses[intent['tag']] = intent['responses']
    
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)
y = tags

model = LogisticRegression()
model.fit(X, y)

def chatbot_response(user_input):
    user_input_vectorized = vectorizer.transform([user_input])
    tag = model.predict(user_input_vectorized)[0]
    return random.choice(responses[tag])

st.title("Ai Chatbot in your service")
st.write("Hi, I am here to assist you . Feel free to ask me anything!")
user_input = st.text_input("You:", "")

if user_input:
    bot_response = chatbot_response(user_input)
    st.write(f"Chatbot: {bot_response}")