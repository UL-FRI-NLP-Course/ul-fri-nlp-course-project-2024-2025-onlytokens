#load /Users/carbs/ul-fri-nlp-course-project-2024-2025-onlytokens/eval/final_results/aggregated_grades_complete.jsonl
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load the data
print("Loading aggregated results...")
data = pd.read_json('aggregated_grades_complete.jsonl', lines=True)

print(f"Total questions evaluated: {len(data)}")
print(f"Columns: {data.columns.tolist()}")

# Create figure with multiple subplots
fig = plt.figure(figsize=(20, 16))

# Updated experiment names
experiments = ['Plain LLM', 'Ours', 'Ours + Quality Filter']
exp_names_short = ['Plain LLM', 'Ours', 'Ours + Quality']

# 1. Overall accuracy comparison (A grades)
ax1 = plt.subplot(2, 3, 1)
accuracy_scores = [
    (data['grade_llm'] == 'A').mean() * 100,
    (data['grade_normal'] == 'A').mean() * 100,
    (data['grade_quality'] == 'A').mean() * 100
]

bars = ax1.bar(experiments, accuracy_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
ax1.set_title('Overall Accuracy Comparison\n(% of Correct Answers)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_ylim(0, 100)
plt.setp(ax1.get_xticklabels(), rotation=15, ha='right')

# Add value labels on bars
for bar, score in zip(bars, accuracy_scores):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{score:.1f}%', ha='center', va='bottom', fontweight='bold')

# 2. Grade distribution for each experiment
ax2 = plt.subplot(2, 3, 2)
grade_data = {
    'Plain LLM': data['grade_llm'].value_counts(),
    'Ours': data['grade_normal'].value_counts(),
    'Ours + Quality': data['grade_quality'].value_counts()
}

x = np.arange(3)  # A, B, C
width = 0.25

# Updated grade labels
grades = ['A', 'B', 'C']
grade_labels = ['Correct', 'Incorrect', 'Not Attempted']
colors = ['#2ECC40', '#FF851B', '#FF4136']

for i, (grade, label) in enumerate(zip(grades, grade_labels)):
    values = [grade_data[exp].get(grade, 0) for exp in ['Plain LLM', 'Ours', 'Ours + Quality']]
    ax2.bar(x + i*width, values, width, label=label, color=colors[i])

ax2.set_title('Response Quality Distribution by Experiment', fontsize=14, fontweight='bold')
ax2.set_xlabel('Experiment', fontsize=12)
ax2.set_ylabel('Number of Questions', fontsize=12)
ax2.set_xticks(x + width)
ax2.set_xticklabels(exp_names_short)
ax2.legend()

# 3. Performance improvement matrix
ax3 = plt.subplot(2, 3, 3)
improvement_matrix = np.zeros((3, 3))
experiments_list = ['grade_llm', 'grade_normal', 'grade_quality']

for i, exp1 in enumerate(experiments_list):
    for j, exp2 in enumerate(experiments_list):
        if i != j:
            # Count how many questions improved from exp1 to exp2
            improved = 0
            for _, row in data.iterrows():
                if (row[exp1] == 'C' and row[exp2] in ['A', 'B']) or \
                   (row[exp1] == 'B' and row[exp2] == 'A'):
                    improved += 1
            improvement_matrix[i][j] = improved

sns.heatmap(improvement_matrix, annot=True, fmt='.0f', cmap='YlOrRd', 
            xticklabels=exp_names_short, yticklabels=exp_names_short, ax=ax3)
ax3.set_title('Questions Improved\n(from row to column)', fontsize=14, fontweight='bold')

# 4. Detailed performance metrics
ax4 = plt.subplot(2, 3, 4)
metrics = []
for exp, name in zip(['grade_llm', 'grade_normal', 'grade_quality'], 
                     exp_names_short):
    grade_counts = data[exp].value_counts()
    total = len(data)
    
    correct = grade_counts.get('A', 0) / total * 100
    incorrect = grade_counts.get('B', 0) / total * 100
    not_attempted = grade_counts.get('C', 0) / total * 100
    
    metrics.append([correct, incorrect, not_attempted])

metrics = np.array(metrics)
x = np.arange(3)
width = 0.6

bottom1 = np.zeros(3)
bottom2 = metrics[:, 0]

p1 = ax4.bar(x, metrics[:, 0], width, label='Correct', color='#2ECC40')
p2 = ax4.bar(x, metrics[:, 1], width, bottom=bottom2, label='Incorrect', color='#FF851B')
p3 = ax4.bar(x, metrics[:, 2], width, bottom=bottom2 + metrics[:, 1], 
             label='Not Attempted', color='#FF4136')

ax4.set_title('Performance Breakdown by Category', fontsize=14, fontweight='bold')
ax4.set_ylabel('Percentage (%)', fontsize=12)
ax4.set_xticks(x)
ax4.set_xticklabels(exp_names_short)
ax4.legend()
ax4.set_ylim(0, 100)

# 5. Agreement analysis between experiments
ax5 = plt.subplot(2, 3, 5)
agreement_data = []
agreement_labels = []

# Plain LLM vs Ours
agreement = (data['grade_llm'] == data['grade_normal']).mean() * 100
agreement_data.append(agreement)
agreement_labels.append('Plain LLM vs\nOurs')

# Plain LLM vs Ours + Quality
agreement = (data['grade_llm'] == data['grade_quality']).mean() * 100
agreement_data.append(agreement)
agreement_labels.append('Plain LLM vs\nOurs + Quality')

# Ours vs Ours + Quality
agreement = (data['grade_normal'] == data['grade_quality']).mean() * 100
agreement_data.append(agreement)
agreement_labels.append('Ours vs\nOurs + Quality')

bars = ax5.bar(agreement_labels, agreement_data, color=['#9B59B6', '#E74C3C', '#F39C12'])
ax5.set_title('Agreement Between Experiments\n(% Same Grade)', fontsize=14, fontweight='bold')
ax5.set_ylabel('Agreement (%)', fontsize=12)
ax5.set_ylim(0, 100)

# Add value labels
for bar, score in zip(bars, agreement_data):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{score:.1f}%', ha='center', va='bottom', fontweight='bold')

# 6. Performance trend analysis
ax6 = plt.subplot(2, 3, 6)
# Convert grades to numeric scores for trend analysis
def grade_to_score(grade):
    return {'A': 2, 'B': 1, 'C': 0}[grade]

data['score_llm'] = data['grade_llm'].apply(grade_to_score)
data['score_normal'] = data['grade_normal'].apply(grade_to_score)
data['score_quality'] = data['grade_quality'].apply(grade_to_score)

# Calculate rolling averages
window_size = 50
data['rolling_llm'] = data['score_llm'].rolling(window=window_size).mean()
data['rolling_normal'] = data['score_normal'].rolling(window=window_size).mean()
data['rolling_quality'] = data['score_quality'].rolling(window=window_size).mean()

x_range = range(window_size, len(data))
ax6.plot(x_range, data['rolling_llm'][window_size:], label='Plain LLM', linewidth=2, color='#FF6B6B')
ax6.plot(x_range, data['rolling_normal'][window_size:], label='Ours', linewidth=2, color='#4ECDC4')
ax6.plot(x_range, data['rolling_quality'][window_size:], label='Ours + Quality', linewidth=2, color='#45B7D1')

ax6.set_title(f'Performance Trends\n(Rolling Average, window={window_size})', fontsize=14, fontweight='bold')
ax6.set_xlabel('Question Number', fontsize=12)
ax6.set_ylabel('Average Score', fontsize=12)
ax6.legend()
ax6.set_ylim(0, 2)

plt.tight_layout()
plt.savefig('experiment_results_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig('experiment_results_analysis.pdf', dpi=300, bbox_inches='tight')

# Create individual plots for the report
# Plot 1: Overall Accuracy Comparison
fig1, ax1 = plt.subplots(figsize=(10, 6))
bars = ax1.bar(experiments, accuracy_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
ax1.set_title('Overall Accuracy Comparison', fontsize=16, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=14)
ax1.set_ylim(0, 100)
plt.setp(ax1.get_xticklabels(), rotation=15, ha='right', fontsize=12)

# Add value labels on bars
for bar, score in zip(bars, accuracy_scores):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{score:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('accuracy_comparison.pdf', dpi=300, bbox_inches='tight')

# Plot 2: Grade Distribution
fig2, ax2 = plt.subplots(figsize=(10, 6))
x = np.arange(3)
width = 0.25

for i, (grade, label) in enumerate(zip(grades, grade_labels)):
    values = [grade_data[exp].get(grade, 0) for exp in ['Plain LLM', 'Ours', 'Ours + Quality']]
    ax2.bar(x + i*width, values, width, label=label, color=colors[i])

ax2.set_title('Response Quality Distribution by Experiment', fontsize=16, fontweight='bold')
ax2.set_xlabel('Experiment', fontsize=14)
ax2.set_ylabel('Number of Questions', fontsize=14)
ax2.set_xticks(x + width)
ax2.set_xticklabels(exp_names_short, fontsize=12)
ax2.legend(fontsize=12)

plt.tight_layout()
plt.savefig('grade_distribution.pdf', dpi=300, bbox_inches='tight')

print("Plots saved successfully!")

# Print detailed statistics
print("\n" + "="*80)
print("DETAILED EXPERIMENT ANALYSIS")
print("="*80)

for exp, name in zip(['grade_llm', 'grade_normal', 'grade_quality'], 
                     ['Plain LLM (No RAG)', 'Ours (RAG Search Pipeline)', 'Ours + Quality Filter']):
    print(f"\n{name}:")
    grade_counts = data[exp].value_counts()
    total = len(data)
    
    print(f"  Correct (A):       {grade_counts.get('A', 0):4d} ({grade_counts.get('A', 0)/total*100:5.1f}%)")
    print(f"  Incorrect (B):     {grade_counts.get('B', 0):4d} ({grade_counts.get('B', 0)/total*100:5.1f}%)")
    print(f"  Not Attempted (C): {grade_counts.get('C', 0):4d} ({grade_counts.get('C', 0)/total*100:5.1f}%)")

print(f"\nTotal questions: {len(data)}")

# Best performing questions (all A's)
all_correct = data[(data['grade_llm'] == 'A') & 
                   (data['grade_normal'] == 'A') & 
                   (data['grade_quality'] == 'A')]
print(f"\nQuestions where all experiments got correct answers: {len(all_correct)} ({len(all_correct)/len(data)*100:.1f}%)")

# Worst performing questions (all C's)
all_failed = data[(data['grade_llm'] == 'C') & 
                  (data['grade_normal'] == 'C') & 
                  (data['grade_quality'] == 'C')]
print(f"Questions where all experiments failed: {len(all_failed)} ({len(all_failed)/len(data)*100:.1f}%)")

# Questions where quality pipeline helped
quality_helped = data[(data['grade_normal'] != 'A') & (data['grade_quality'] == 'A')]
print(f"Questions where Quality Filter improved to correct: {len(quality_helped)} ({len(quality_helped)/len(data)*100:.1f}%)")

# Questions where our system helped
rag_helped = data[(data['grade_llm'] != 'A') & (data['grade_normal'] == 'A')]
print(f"Questions where our system improved to correct: {len(rag_helped)} ({len(rag_helped)/len(data)*100:.1f}%)")

print("\n" + "="*80)

