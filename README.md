# Customer-Segmentation
Mall Customer Segmentation using K means clustering

A machine learning-based web application that segments customers based on their income and spending score using K-Means clustering.

## Features
- Predicts customer segments based on input data
- Provides business-oriented recommendations
- Visualizes customer distribution using graphs
- Maintains a history of recent predictions

## Tech Stack
- Python
- Flask
- Machine Learning (K-Means Clustering)
- HTML, CSS

## How It Works
1. The user inputs income and spending score
2. The trained K-Means model predicts the customer cluster
3. The system displays:
   - Customer segment
   - Recommendation
   - Graph visualization

## Project Structure
app.py
templates/
static/
customer_segmentation.pkl

## Running the Application
```bash
pip install -r requirements.txt
python app.py

Example Output
High Value Customers: Target with premium offerings
Low Value Customers: Focus on cost-effective strategies

Author

Divyanshu Shukla
Diya Attri
Faijul Rahman
Fareha Farat
