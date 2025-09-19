import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("COMPREHENSIVE DATA ANALYSIS PROJECT")
print("Dataset: Iris Dataset")
print("=" * 60)

# ==========================================
# TASK 1: LOAD AND EXPLORE THE DATASET
# ==========================================

print("\n" + "="*50)
print("TASK 1: LOAD AND EXPLORE THE DATASET")
print("="*50)

try:
    # Load the Iris dataset
    iris_data = load_iris()
    
    # Create a DataFrame
    df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    df['species'] = iris_data.target
    df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    print("✅ Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    
    # Display the first few rows
    print("\n📊 First 5 rows of the dataset:")
    print(df.head())
    
    # Explore the structure of the dataset
    print("\n📋 Dataset Information:")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    
    print("\n📊 Data types:")
    print(df.dtypes)
    
    print("\n🔍 Column names:")
    for i, col in enumerate(df.columns):
        print(f"{i+1}. {col}")
    
    # Check for missing values
    print("\n❓ Missing values check:")
    missing_values = df.isnull().sum()
    print(missing_values)
    
    if missing_values.sum() == 0:
        print("✅ No missing values found in the dataset!")
    else:
        print("⚠️ Found missing values - cleaning required")
        # Fill missing values (if any) - using forward fill as example
        df = df.fillna(method='ffill')
        print("✅ Missing values handled using forward fill")
    
except Exception as e:
    print(f"❌ Error loading dataset: {e}")

# ==========================================
# TASK 2: BASIC DATA ANALYSIS
# ==========================================

print("\n" + "="*50)
print("TASK 2: BASIC DATA ANALYSIS")
print("="*50)

try:
    # Compute basic statistics for numerical columns
    print("\n📈 Basic Statistics for Numerical Columns:")
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    print(df[numerical_columns].describe())
    
    # Group by species and compute mean values
    print("\n🔬 Analysis by Species:")
    species_analysis = df.groupby('species_name')[numerical_columns].mean()
    print(species_analysis)
    
    # Additional analysis - standard deviation by species
    print("\n📊 Standard Deviation by Species:")
    species_std = df.groupby('species_name')[numerical_columns].std()
    print(species_std)
    
    # Find interesting patterns
    print("\n🔍 INTERESTING FINDINGS:")
    
    # Find the species with largest average measurements
    max_sepal_length = species_analysis['sepal length (cm)'].idxmax()
    max_petal_length = species_analysis['petal length (cm)'].idxmax()
    
    print(f"• Species with largest average sepal length: {max_sepal_length}")
    print(f"• Species with largest average petal length: {max_petal_length}")
    
    # Calculate correlations
    correlation_matrix = df[numerical_columns].corr()
    print(f"• Strongest positive correlation: {correlation_matrix.unstack().drop_duplicates().sort_values(ascending=False).iloc[1]:.3f}")
    
    # Species distribution
    print(f"• Dataset is perfectly balanced: {df['species_name'].value_counts().to_dict()}")
    
except Exception as e:
    print(f"❌ Error in basic analysis: {e}")

# ==========================================
# TASK 3: DATA VISUALIZATION
# ==========================================

print("\n" + "="*50)
print("TASK 3: DATA VISUALIZATION")
print("="*50)

# Set up the plotting style
plt.style.use('default')
sns.set_palette("husl")

# Create a figure with subplots
fig = plt.figure(figsize=(16, 12))

try:
    # 1. LINE CHART - Trends across samples (simulating time series)
    plt.subplot(2, 2, 1)
    
    # Create a line chart showing how measurements change across samples
    sample_indices = df.index
    for species in df['species_name'].unique():
        species_data = df[df['species_name'] == species]
        plt.plot(species_data.index, species_data['sepal length (cm)'], 
                marker='o', alpha=0.7, label=f'{species} - Sepal Length', linewidth=2)
    
    plt.title('Sepal Length Trends Across Samples', fontsize=14, fontweight='bold')
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Sepal Length (cm)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. BAR CHART - Average petal length per species
    plt.subplot(2, 2, 2)
    
    avg_petal_length = df.groupby('species_name')['petal length (cm)'].mean()
    bars = plt.bar(avg_petal_length.index, avg_petal_length.values, 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
    
    plt.title('Average Petal Length by Species', fontsize=14, fontweight='bold')
    plt.xlabel('Species', fontsize=12)
    plt.ylabel('Average Petal Length (cm)', fontsize=12)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45)
    
    # 3. HISTOGRAM - Distribution of sepal width
    plt.subplot(2, 2, 3)
    
    plt.hist(df['sepal width (cm)'], bins=20, color='#96CEB4', alpha=0.7, edgecolor='black')
    plt.axvline(df['sepal width (cm)'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {df["sepal width (cm)"].mean():.2f}')
    
    plt.title('Distribution of Sepal Width', fontsize=14, fontweight='bold')
    plt.xlabel('Sepal Width (cm)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. SCATTER PLOT - Relationship between sepal and petal length
    plt.subplot(2, 2, 4)
    
    colors = {'setosa': '#FF6B6B', 'versicolor': '#4ECDC4', 'virginica': '#45B7D1'}
    
    for species in df['species_name'].unique():
        species_data = df[df['species_name'] == species]
        plt.scatter(species_data['sepal length (cm)'], species_data['petal length (cm)'],
                   c=colors[species], label=species, alpha=0.7, s=60)
    
    plt.title('Sepal Length vs Petal Length', fontsize=14, fontweight='bold')
    plt.xlabel('Sepal Length (cm)', fontsize=12)
    plt.ylabel('Petal Length (cm)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(df['sepal length (cm)'], df['petal length (cm)'], 1)
    p = np.poly1d(z)
    plt.plot(df['sepal length (cm)'], p(df['sepal length (cm)']), "r--", alpha=0.8, linewidth=2)
    
    plt.tight_layout()
    plt.show()
    
    print("✅ All visualizations created successfully!")
    
except Exception as e:
    print(f"❌ Error creating visualizations: {e}")

# ==========================================
# ADDITIONAL ANALYSIS
# ==========================================

print("\n" + "="*50)
print("ADDITIONAL INSIGHTS")
print("="*50)

try:
    # Create a correlation heatmap
    plt.figure(figsize=(10, 8))
    
    # Calculate correlation matrix
    corr_matrix = df[numerical_columns].corr()
    
    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
    
    plt.title('Correlation Matrix of Iris Features', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
    
    # Summary statistics by species
    print("\n📊 SUMMARY BY SPECIES:")
    for species in df['species_name'].unique():
        species_data = df[df['species_name'] == species]
        print(f"\n{species.upper()}:")
        print(f"  • Count: {len(species_data)}")
        print(f"  • Avg Sepal Length: {species_data['sepal length (cm)'].mean():.2f} cm")
        print(f"  • Avg Petal Length: {species_data['petal length (cm)'].mean():.2f} cm")
        print(f"  • Sepal L/W Ratio: {(species_data['sepal length (cm)'] / species_data['sepal width (cm)']).mean():.2f}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("Key Findings:")
    print("• The Iris dataset contains 150 samples of 3 species (50 each)")
    print("• Virginica has the largest average measurements")
    print("• Strong positive correlation between petal length and sepal length")
    print("• Each species shows distinct clustering patterns")
    print("• No missing values - clean dataset")
    
except Exception as e:
    print(f"❌ Error in additional analysis: {e}")

# Error handling demonstration
print("\n🔧 ERROR HANDLING DEMONSTRATION:")
try:
    # Attempt to access a non-existent column
    df['non_existent_column'].mean()
except KeyError as e:
    print(f"✅ Handled KeyError: Column not found - {e}")

try:
    # Attempt to divide by zero
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"✅ Handled ZeroDivisionError: {e}")

print("\n✅ All tasks completed successfully with proper error handling!")