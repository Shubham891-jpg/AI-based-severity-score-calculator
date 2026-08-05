import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import sys
from typing import List, Dict

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.predict import SeverityPredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation for IT ticket severity prediction."""
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize evaluator with trained model.
        
        Args:
            model_dir: Directory containing trained model files
        """
        self.predictor = SeverityPredictor(model_dir)
        self.evaluation_results = {}
        
    def load_test_data(self, test_data_path: str = "data/processed/test.csv") -> pd.DataFrame:
        """
        Load test dataset.
        
        Args:
            test_data_path: Path to test data CSV file
            
        Returns:
            Test dataset DataFrame
        """
        try:
            if not os.path.exists(test_data_path):
                # If processed test data doesn't exist, use raw data
                logger.warning(f"Test data not found at {test_data_path}, using raw data")
                test_data_path = "data/raw/tickets_raw.csv"
                
            df = pd.read_csv(test_data_path)
            logger.info(f"Loaded {len(df)} test samples from {test_data_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load test data: {str(e)}")
            raise
    
    def evaluate_predictions(self, test_df: pd.DataFrame) -> Dict:
        """
        Evaluate model predictions on test data.
        
        Args:
            test_df: Test dataset
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            logger.info("Starting model evaluation...")
            
            # Get predictions
            ticket_texts = test_df['ticket_text'].tolist()
            true_scores = test_df['severity_score'].values
            
            predictions = self.predictor.predict_batch(ticket_texts)
            predicted_scores = np.array([pred['severity_score'] for pred in predictions])
            
            # Calculate metrics
            metrics = self._calculate_metrics(true_scores, predicted_scores)
            
            # Add prediction details
            metrics['predictions'] = predictions
            metrics['true_scores'] = true_scores.tolist()
            metrics['predicted_scores'] = predicted_scores.tolist()
            
            # Analyze errors
            errors = predicted_scores - true_scores
            metrics['errors'] = {
                'mean_error': float(np.mean(errors)),
                'std_error': float(np.std(errors)),
                'max_error': float(np.max(np.abs(errors))),
                'error_distribution': self._analyze_error_distribution(errors)
            }
            
            # Analyze by severity category
            metrics['category_analysis'] = self._analyze_by_category(
                true_scores, predicted_scores, predictions
            )
            
            # Analyze by language
            metrics['language_analysis'] = self._analyze_by_language(
                test_df, predictions, true_scores, predicted_scores
            )
            
            self.evaluation_results = metrics
            logger.info("Model evaluation completed")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            raise
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate regression metrics."""
        return {
            'mse': float(mean_squared_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'r2': float(r2_score(y_true, y_pred)),
            'mean_true': float(np.mean(y_true)),
            'mean_pred': float(np.mean(y_pred)),
            'std_true': float(np.std(y_true)),
            'std_pred': float(np.std(y_pred)),
            'min_true': float(np.min(y_true)),
            'max_true': float(np.max(y_true)),
            'min_pred': float(np.min(y_pred)),
            'max_pred': float(np.max(y_pred))
        }
    
    def _analyze_error_distribution(self, errors: np.ndarray) -> Dict:
        """Analyze distribution of prediction errors."""
        return {
            'within_5': float(np.mean(np.abs(errors) <= 5) * 100),
            'within_10': float(np.mean(np.abs(errors) <= 10) * 100),
            'within_15': float(np.mean(np.abs(errors) <= 15) * 100),
            'within_20': float(np.mean(np.abs(errors) <= 20) * 100),
            'percentiles': {
                '25th': float(np.percentile(np.abs(errors), 25)),
                '50th': float(np.percentile(np.abs(errors), 50)),
                '75th': float(np.percentile(np.abs(errors), 75)),
                '90th': float(np.percentile(np.abs(errors), 90)),
                '95th': float(np.percentile(np.abs(errors), 95))
            }
        }
    
    def _analyze_by_category(self, true_scores: np.ndarray, 
                           predicted_scores: np.ndarray, 
                           predictions: List[Dict]) -> Dict:
        """Analyze performance by severity category."""
        try:
            category_analysis = {}
            
            # Define category ranges
            categories = {
                'Critical': (80, 100),
                'High': (60, 80),
                'Medium': (40, 60),
                'Low': (20, 40),
                'Minimal': (10, 20)
            }
            
            for category, (min_score, max_score) in categories.items():
                # Find samples in this category
                mask = (true_scores >= min_score) & (true_scores < max_score)
                if category == 'Critical':  # Include 100 for Critical
                    mask = (true_scores >= min_score) & (true_scores <= max_score)
                
                if np.sum(mask) > 0:
                    cat_true = true_scores[mask]
                    cat_pred = predicted_scores[mask]
                    
                    category_analysis[category] = {
                        'count': int(np.sum(mask)),
                        'metrics': self._calculate_metrics(cat_true, cat_pred),
                        'accuracy': float(np.mean(
                            [pred['severity_category'] == category 
                             for i, pred in enumerate(predictions) if mask[i]]
                        ) * 100) if np.sum(mask) > 0 else 0.0
                    }
                else:
                    category_analysis[category] = {
                        'count': 0,
                        'metrics': {},
                        'accuracy': 0.0
                    }
            
            return category_analysis
            
        except Exception as e:
            logger.error(f"Category analysis failed: {str(e)}")
            return {}
    
    def _analyze_by_language(self, test_df: pd.DataFrame, 
                           predictions: List[Dict],
                           true_scores: np.ndarray,
                           predicted_scores: np.ndarray) -> Dict:
        """Analyze performance by language."""
        try:
            language_analysis = {}
            
            # Group by detected language
            languages = {}
            for i, pred in enumerate(predictions):
                lang = pred.get('detected_language', 'unknown')
                if lang not in languages:
                    languages[lang] = []
                languages[lang].append(i)
            
            for lang, indices in languages.items():
                if len(indices) > 0:
                    lang_true = true_scores[indices]
                    lang_pred = predicted_scores[indices]
                    
                    language_analysis[lang] = {
                        'count': len(indices),
                        'metrics': self._calculate_metrics(lang_true, lang_pred),
                        'sample_texts': [
                            test_df.iloc[idx]['ticket_text'][:100] + "..."
                            for idx in indices[:3]  # First 3 samples
                        ]
                    }
            
            return language_analysis
            
        except Exception as e:
            logger.error(f"Language analysis failed: {str(e)}")
            return {}
    
    def generate_evaluation_report(self, output_dir: str = "evaluation_results"):
        """
        Generate comprehensive evaluation report.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            if not self.evaluation_results:
                raise ValueError("No evaluation results available. Run evaluate_predictions first.")
            
            # Generate text report
            self._generate_text_report(output_dir)
            
            # Generate visualizations
            self._generate_visualizations(output_dir)
            
            # Save detailed results
            self._save_detailed_results(output_dir)
            
            logger.info(f"Evaluation report generated in {output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to generate evaluation report: {str(e)}")
            raise
    
    def _generate_text_report(self, output_dir: str):
        """Generate text-based evaluation report."""
        report_path = os.path.join(output_dir, "evaluation_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("IT TICKET SEVERITY PREDICTION - EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Overall metrics
            f.write("OVERALL PERFORMANCE METRICS\n")
            f.write("-" * 30 + "\n")
            metrics = self.evaluation_results
            f.write(f"R² Score: {metrics['r2']:.4f}\n")
            f.write(f"RMSE: {metrics['rmse']:.2f}\n")
            f.write(f"MAE: {metrics['mae']:.2f}\n")
            f.write(f"MSE: {metrics['mse']:.2f}\n\n")
            
            # Prediction statistics
            f.write("PREDICTION STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Mean True Score: {metrics['mean_true']:.2f}\n")
            f.write(f"Mean Predicted Score: {metrics['mean_pred']:.2f}\n")
            f.write(f"Std True Score: {metrics['std_true']:.2f}\n")
            f.write(f"Std Predicted Score: {metrics['std_pred']:.2f}\n")
            f.write(f"Score Range (True): [{metrics['min_true']:.1f}, {metrics['max_true']:.1f}]\n")
            f.write(f"Score Range (Pred): [{metrics['min_pred']:.1f}, {metrics['max_pred']:.1f}]\n\n")
            
            # Error analysis
            f.write("ERROR ANALYSIS\n")
            f.write("-" * 15 + "\n")
            errors = metrics['errors']
            f.write(f"Mean Error: {errors['mean_error']:.2f}\n")
            f.write(f"Error Std: {errors['std_error']:.2f}\n")
            f.write(f"Max Absolute Error: {errors['max_error']:.2f}\n\n")
            
            f.write("Error Distribution (% within threshold):\n")
            error_dist = errors['error_distribution']
            f.write(f"  Within ±5 points: {error_dist['within_5']:.1f}%\n")
            f.write(f"  Within ±10 points: {error_dist['within_10']:.1f}%\n")
            f.write(f"  Within ±15 points: {error_dist['within_15']:.1f}%\n")
            f.write(f"  Within ±20 points: {error_dist['within_20']:.1f}%\n\n")
            
            # Category analysis
            f.write("PERFORMANCE BY SEVERITY CATEGORY\n")
            f.write("-" * 35 + "\n")
            for category, analysis in metrics['category_analysis'].items():
                f.write(f"{category}:\n")
                f.write(f"  Count: {analysis['count']}\n")
                if analysis['count'] > 0:
                    cat_metrics = analysis['metrics']
                    f.write(f"  R²: {cat_metrics.get('r2', 0):.4f}\n")
                    f.write(f"  RMSE: {cat_metrics.get('rmse', 0):.2f}\n")
                    f.write(f"  Category Accuracy: {analysis['accuracy']:.1f}%\n")
                f.write("\n")
            
            # Language analysis
            f.write("PERFORMANCE BY LANGUAGE\n")
            f.write("-" * 25 + "\n")
            for language, analysis in metrics['language_analysis'].items():
                f.write(f"{language.upper()}:\n")
                f.write(f"  Count: {analysis['count']}\n")
                lang_metrics = analysis['metrics']
                f.write(f"  R²: {lang_metrics.get('r2', 0):.4f}\n")
                f.write(f"  RMSE: {lang_metrics.get('rmse', 0):.2f}\n")
                f.write(f"  MAE: {lang_metrics.get('mae', 0):.2f}\n\n")
    
    def _generate_visualizations(self, output_dir: str):
        """Generate evaluation visualizations."""
        try:
            plt.style.use('default')
            
            true_scores = np.array(self.evaluation_results['true_scores'])
            predicted_scores = np.array(self.evaluation_results['predicted_scores'])
            errors = predicted_scores - true_scores
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('IT Ticket Severity Prediction - Model Evaluation', fontsize=16)
            
            # 1. Actual vs Predicted scatter plot
            axes[0, 0].scatter(true_scores, predicted_scores, alpha=0.6)
            axes[0, 0].plot([10, 100], [10, 100], 'r--', lw=2)
            axes[0, 0].set_xlabel('True Severity Score')
            axes[0, 0].set_ylabel('Predicted Severity Score')
            axes[0, 0].set_title('Actual vs Predicted Scores')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add R² score to the plot
            r2 = self.evaluation_results['r2']
            axes[0, 0].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[0, 0].transAxes,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # 2. Error distribution histogram
            axes[0, 1].hist(errors, bins=20, alpha=0.7, edgecolor='black')
            axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2)
            axes[0, 1].set_xlabel('Prediction Error (Predicted - True)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Error Distribution')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Residuals plot
            axes[1, 0].scatter(predicted_scores, errors, alpha=0.6)
            axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=2)
            axes[1, 0].set_xlabel('Predicted Severity Score')
            axes[1, 0].set_ylabel('Residuals (Predicted - True)')
            axes[1, 0].set_title('Residuals vs Predicted')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Category performance
            categories = list(self.evaluation_results['category_analysis'].keys())
            category_r2 = [
                self.evaluation_results['category_analysis'][cat]['metrics'].get('r2', 0)
                for cat in categories
            ]
            category_counts = [
                self.evaluation_results['category_analysis'][cat]['count']
                for cat in categories
            ]
            
            # Create bar plot with counts as labels
            bars = axes[1, 1].bar(categories, category_r2, alpha=0.7)
            axes[1, 1].set_ylabel('R² Score')
            axes[1, 1].set_title('Performance by Severity Category')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            # Add count labels on bars
            for bar, count in zip(bars, category_counts):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'n={count}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(output_dir, "evaluation_plots.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Evaluation plots saved to {plot_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate visualizations: {str(e)}")
    
    def _save_detailed_results(self, output_dir: str):
        """Save detailed evaluation results to CSV."""
        try:
            # Create detailed results DataFrame
            results_data = []
            
            predictions = self.evaluation_results['predictions']
            true_scores = self.evaluation_results['true_scores']
            
            for i, (pred, true_score) in enumerate(zip(predictions, true_scores)):
                results_data.append({
                    'index': i,
                    'true_score': true_score,
                    'predicted_score': pred['severity_score'],
                    'error': pred['severity_score'] - true_score,
                    'abs_error': abs(pred['severity_score'] - true_score),
                    'true_category': self.predictor.scaler.get_severity_category(true_score),
                    'predicted_category': pred['severity_category'],
                    'category_match': (
                        self.predictor.scaler.get_severity_category(true_score) == 
                        pred['severity_category']
                    ),
                    'confidence': pred.get('confidence', 0),
                    'detected_language': pred.get('detected_language', 'unknown')
                })
            
            results_df = pd.DataFrame(results_data)
            
            # Save to CSV
            csv_path = os.path.join(output_dir, "detailed_results.csv")
            results_df.to_csv(csv_path, index=False)
            
            logger.info(f"Detailed results saved to {csv_path}")
            
        except Exception as e:
            logger.error(f"Failed to save detailed results: {str(e)}")
    
    def print_summary(self):
        """Print evaluation summary to console."""
        if not self.evaluation_results:
            print("No evaluation results available. Run evaluate_predictions first.")
            return
        
        metrics = self.evaluation_results
        
        print("\n" + "="*60)
        print("MODEL EVALUATION SUMMARY")
        print("="*60)
        print(f"R² Score: {metrics['r2']:.4f}")
        print(f"RMSE: {metrics['rmse']:.2f}")
        print(f"MAE: {metrics['mae']:.2f}")
        print(f"Mean Prediction: {metrics['mean_pred']:.2f}")
        print(f"Prediction Range: [{metrics['min_pred']:.1f}, {metrics['max_pred']:.1f}]")
        
        print(f"\nError Analysis:")
        error_dist = metrics['errors']['error_distribution']
        print(f"  Within ±10 points: {error_dist['within_10']:.1f}%")
        print(f"  Within ±15 points: {error_dist['within_15']:.1f}%")
        
        print(f"\nLanguage Performance:")
        for lang, analysis in metrics['language_analysis'].items():
            print(f"  {lang.upper()}: {analysis['count']} samples, "
                  f"R² = {analysis['metrics'].get('r2', 0):.4f}")
        
        print("="*60)

def main():
    """Main evaluation function."""
    try:
        evaluator = ModelEvaluator()
        
        # Load test data
        test_df = evaluator.load_test_data()
        
        # Run evaluation
        results = evaluator.evaluate_predictions(test_df)
        
        # Print summary
        evaluator.print_summary()
        
        # Generate full report
        evaluator.generate_evaluation_report()
        
        print("\nEvaluation completed! Check 'evaluation_results' directory for detailed reports.")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()