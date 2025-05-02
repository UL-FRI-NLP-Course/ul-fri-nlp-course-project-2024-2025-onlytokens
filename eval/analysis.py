import json
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from collections import Counter, defaultdict

def load_results(file_path):
    results = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                # Convert A/B/C to Correct/Incorrect/Not Attempted at load time
                if 'final_grade' in data:
                    if isinstance(data['final_grade'], str):
                        if data['final_grade'] == 'A':
                            data['final_grade'] = 'Correct'
                        elif data['final_grade'] == 'B':
                            data['final_grade'] = 'Incorrect'
                        elif data['final_grade'] == 'C':
                            data['final_grade'] = 'Not Attempted'
                    elif isinstance(data['final_grade'], (int, float)):
                        data['final_grade'] = 'Correct' if data['final_grade'] == 1 else 'Incorrect' if data['final_grade'] == 0 else 'Not Attempted'
                results.append(data)
            except json.JSONDecodeError:
                continue
    return results

def create_sunburst_chart(data, title, id_suffix=""):
    # Create sunburst chart
    fig = go.Figure(go.Sunburst(
        labels=data['labels'],
        parents=data['parents'],
        values=data['values'],
        branchvalues="total",
        marker=dict(colors=data['colors']),  # Use manual colors
        textfont=dict(size=12), # Slightly smaller font for potentially longer labels
        insidetextorientation='radial' # Adjust text orientation if needed
    ))
    
    fig.update_layout(
        title={
            'text': title,
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        },
        width=1200,
        height=1000,
        template="plotly_white"
    )
    
    fig.update_traces(
        hovertemplate="""
        <b>%{label}</b><br>
        Count: %{value}<br>
        Percentage of parent: %{percentParent:.1f}%<br>
        Percentage of total: %{percentRoot:.1f}%
        <extra></extra>
        """
    )
    
    # Save as PDF with high DPI
    fig.write_image("/Users/carbs/ul-fri-nlp-course-project-2024-2025-onlytokens/report/fig/distribution" + id_suffix + ".pdf", scale=2)
    
    # Also display the figure
    fig.show()
    
    return fig

def analyze_performance(results):
    # Create data structures for both charts
    grade_topic_data = defaultdict(lambda: defaultdict(int))
    grade_answer_type_data = defaultdict(lambda: defaultdict(int))
    
    # Define consistent and vibrant color schemes
    grade_colors = {
        'Correct': '#2ECC71',       # Emerald Green
        'Incorrect': '#E74C3C',     # Alizarin Red
        'Not Attempted': '#F39C12'  # Sunflower Orange
    }
    
    topic_colors = {
        'Politics': '#3498DB',      # Peter River Blue
        'Science and technology': '#1ABC9C',  # Turquoise
        'Art': '#9B59B6',           # Amethyst Purple
        'Music': '#F1C40F',         # Sun Flower Yellow
        'Sports': '#E67E22',        # Carrot Orange
        'TV shows': '#2980B9',      # Belize Hole Blue
        'Video games': '#27AE60',    # Nephritis Green
        'History': '#C0392B',       # Pomegranate Red
        'Geography': '#8E44AD',     # Wisteria Purple
        'Other': '#BDC3C7',         # Silver
        'Unknown': '#7F8C8D'        # Asbestos Gray
    }
    
    answer_type_colors = {
        'Factual': '#3498DB',      # Peter River Blue
        'Opinion': '#1ABC9C',      # Turquoise
        'Analysis': '#9B59B6',     # Amethyst Purple
        'List': '#F1C40F',         # Sun Flower Yellow
        'Description': '#E67E22',  # Carrot Orange
        'Number': '#2980B9',       # Belize Hole Blue
        'Date': '#27AE60',         # Nephritis Green
        'Person': '#C0392B',       # Pomegranate Red
        'Place': '#8E44AD',        # Wisteria Purple
        'Other': '#BDC3C7',         # Silver
        'Unknown': '#7F8C8D'        # Asbestos Gray
    }
    
    # Get all unique topics and answer types first
    all_topics = set()
    all_answer_types = set()
    for result in results:
        all_topics.add(result.get('metadata', {}).get('topic', 'Unknown'))
        all_answer_types.add(result.get('metadata', {}).get('answer_type', 'Unknown'))
        
    # Process results and populate data structures
    for result in results:
        grade = result.get('final_grade', 'Unknown')
        topic = result.get('metadata', {}).get('topic', 'Unknown')
        answer_type = result.get('metadata', {}).get('answer_type', 'Unknown')
        
        grade_topic_data[grade][topic] += 1
        grade_answer_type_data[grade][answer_type] += 1
    
    # Prepare data for topic distribution
    topic_data = {
        'labels': ['All'],
        'parents': [''],
        'values': [len(results)],
        'colors': ['#FFFFFF'] # Root color
    }
    
    for grade in grade_colors: # Iterate through defined grades to ensure order
        if grade in grade_topic_data: # Check if grade exists in data
            topic_data['labels'].append(grade)
            topic_data['parents'].append('All')
            topic_data['values'].append(sum(grade_topic_data[grade].values()))
            topic_data['colors'].append(grade_colors.get(grade, '#7F8C8D'))
            
            for topic in all_topics: # Iterate through all possible topics
                count = grade_topic_data[grade].get(topic, 0)
                if count > 0: # Only add if count is greater than 0
                    # Ensure unique labels by including the grade
                    topic_data['labels'].append(f'{topic} ({grade})')
                    topic_data['parents'].append(grade)
                    topic_data['values'].append(count)
                    topic_data['colors'].append(topic_colors.get(topic, '#7F8C8D'))
    
    # Prepare data for answer type distribution
    answer_type_data = {
        'labels': ['All'],
        'parents': [''],
        'values': [len(results)],
        'colors': ['#FFFFFF'] # Root color
    }
    
    for grade in grade_colors: # Iterate through defined grades
        if grade in grade_answer_type_data:
            answer_type_data['labels'].append(grade)
            answer_type_data['parents'].append('All')
            answer_type_data['values'].append(sum(grade_answer_type_data[grade].values()))
            answer_type_data['colors'].append(grade_colors.get(grade, '#7F8C8D'))
            
            for answer_type in all_answer_types: # Iterate through all possible answer types
                count = grade_answer_type_data[grade].get(answer_type, 0)
                if count > 0:
                    # Ensure unique labels by including the grade
                    answer_type_data['labels'].append(f'{answer_type} ({grade})') 
                    answer_type_data['parents'].append(grade)
                    answer_type_data['values'].append(count)
                    answer_type_data['colors'].append(answer_type_colors.get(answer_type, '#7F8C8D'))
    
    # Create visualizations
    fig_topics = create_sunburst_chart(topic_data, "Performance by Topic", "_topics")
    fig_answer_types = create_sunburst_chart(answer_type_data, "Performance by Answer Type", "_answer_types")
    
    # Save the answer types distribution plot
    fig_answer_types.write_image("report/fig/answer_types_distribution.png", scale=2)

    # Save the topics distribution plot 
    fig_topics.write_image("report/fig/topics_distribution.png", scale=2)
    
    # Print analysis
    print("\n=== Model Performance Analysis ===\n")
    
    # Overall statistics
    total_questions = len(results)
    total_correct = sum(grade_topic_data['Correct'].values())
    overall_accuracy = (total_correct / total_questions) * 100
    
    print(f"Overall Performance:")
    print(f"Total questions: {total_questions}")
    print(f"Total correct: {total_correct}")
    print(f"Overall accuracy: {overall_accuracy:.1f}%\n")
    
    # Topic-wise analysis
    print("Performance by Topic:")
    topic_metrics = []
    for topic in sorted(all_topics):
        total_topic = sum(grade_topic_data[grade].get(topic, 0) for grade in grade_topic_data)
        if total_topic > 0:
            correct = grade_topic_data['Correct'].get(topic, 0)
            incorrect = grade_topic_data['Incorrect'].get(topic, 0)
            not_attempted = grade_topic_data['Not Attempted'].get(topic, 0)
            accuracy = (correct / total_topic) * 100
            topic_metrics.append({
                'topic': topic,
                'total': total_topic,
                'correct': correct,
                'accuracy': accuracy
            })
            
            print(f"\n{topic}:")
            print(f"  Total questions: {total_topic}")
            print(f"  Correct: {correct} ({accuracy:.1f}%)")
            print(f"  Incorrect: {incorrect} ({(incorrect/total_topic)*100:.1f}%)")
            print(f"  Not attempted: {not_attempted} ({(not_attempted/total_topic)*100:.1f}%)")
    
    # Answer type analysis
    print("\nPerformance by Answer Type:")
    answer_type_metrics = []
    for answer_type in sorted(all_answer_types):
        total = sum(grade_answer_type_data[grade].get(answer_type, 0) for grade in grade_answer_type_data)
        if total > 0:
            correct = grade_answer_type_data['Correct'].get(answer_type, 0)
            accuracy = (correct / total) * 100
            answer_type_metrics.append({
                'type': answer_type,
                'total': total,
                'correct': correct,
                'accuracy': accuracy
            })
            
            print(f"\n{answer_type}:")
            print(f"  Total questions: {total}")
            print(f"  Correct: {correct} ({accuracy:.1f}%)")

def main():
    results = load_results('eval/gpt4o_results.jsonl')
    analyze_performance(results)

if __name__ == "__main__":
    main()
