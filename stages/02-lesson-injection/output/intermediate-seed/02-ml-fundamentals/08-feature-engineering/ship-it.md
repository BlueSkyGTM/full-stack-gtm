## Ship It

To deploy this in a production GTM stack, you need the feature pipeline serialized and replayable. The engineered features that survived selection define your enrichment schema. The transforms (log, scaling, binning thresholds) need to be applied identically to every new prospect record at inference time.

```python
import json
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer

numeric_features = ['log_revenue', 'employee_count_scaled', 'tech_stack_count']
categorical_features = ['industry']

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
])

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])

X_deploy = df[['revenue', 'employee_count', 'industry', 'tech_stack_count']].copy()
X_deploy['log_revenue'] = np.log1p(X_deploy['revenue'])
X_deploy['employee_count_scaled'] = np.log1p(X_deploy['employee_count'])

full_pipeline.fit(X_deploy[['log_revenue', 'employee_count_scaled', 'tech_stack_count', 'industry']], df['converted'])

with open('lead_score_pipeline.pkl', 'wb') as f:
    pickle.dump(full_pipeline, f)

schema = {
    'version': '1.0',
    'features': {
        'log_revenue': {'source': 'revenue', 'transform': 'log1p'},
        'employee_count_scaled': {'source': 'employee_count', 'transform': 'log1p + standardize'},
        'tech_stack_count': {'source': 'tech_stack_count', 'transform': 'none'},
        'industry': {'source': 'industry', 'transform': 'one_hot_drop_first'}
    },
    'selected_features': selected,
    'model': 'RandomForestClassifier',
    'training_rows': len(df),
    'conversion_rate': float(df['converted'].mean())
}

with open('enrichment_schema.json', 'w') as f:
    json.dump(schema, f, indent=2)

print("=== DEPLOYMENT ARTIFACTS ===")
print("Saved: lead_score_pipeline.pkl")
print("Saved: enrichment_schema.json")
print(f"\nSchema:")
print(json.dumps(schema, indent=2))

new_prospect = pd.DataFrame([{
    'log_revenue': np.log1p(3_500_000),
    'employee_count_scaled': np.log1p(120),
    'tech_stack_count': 14,
    'industry': 'SaaS'
}])

score = full_pipeline.predict_proba(new_prospect)[0][1]
print(f"\n=== INFERENCE TEST ===")
print(f"New prospect: SaaS, $3.5M revenue, 120 employees, 14 tools")
print(f"Conversion probability: {score:.3f}")
```

The `enrichment_schema.json` file is your contract with the enrichment layer. When you configure data enrichment in a Clay table or a similar tool, these are the columns you pull. If a column isn't in `selected_features`, it doesn't get enriched. This is how you cut enrichment costs — you stop paying your data provider for columns that permutation importance already told you carry no signal. In a Clay waterfall configuration, this schema defines which providers you call and which fields you request from each: if revenue comes from ZoomInfo at a lower cost than LinkedIn and both have the same predictive value, the waterfall calls ZoomInfo first.

The pipeline pickle is your inference artifact. Load it in whatever runtime scores new leads — a scheduled batch job, an API endpoint, or a webhook handler. The critical rule: the exact same transforms applied during training must be applied at inference. The `ColumnTransformer` and `Pipeline` objects handle this — they store the fitted parameters (scaling means, one-hot categories) and apply them to new data. Never recompute transforms on inference data.