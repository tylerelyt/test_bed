#!/usr/bin/env python3
"""检查关键依赖版本"""

packages = [
    'numpy',
    'scikit-learn', 
    'pandas',
    'tensorflow',
    'torch',
    'transformers',
    'gradio',
    'keras',
    'tf-keras'
]

print("📦 依赖版本检查")
print("="*50)

for package in packages:
    try:
        if package == 'scikit-learn':
            import sklearn
            version = sklearn.__version__
        elif package == 'tf-keras':
            import tf_keras
            version = tf_keras.__version__
        else:
            mod = __import__(package)
            version = mod.__version__
        
        print(f"✅ {package:20s} : {version}")
    except Exception as e:
        print(f"❌ {package:20s} : {str(e)[:50]}")

print("="*50)

