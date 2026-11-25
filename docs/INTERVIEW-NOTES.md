# Interview preparation

A factual opening: “This is a small PyTorch image-classification portfolio project. It compares a neural network with a linear baseline, keeps the official test set separate, and makes the mistakes inspectable. I used AI assistance and am studying and reproducing the implementation.” Change the final sentence only as your actual work and understanding change.

Be ready to explain:

1. Why 64 inputs and 10 outputs? An 8×8 prepared image has 64 pixel values; the target classes are digits 0–9.
2. Why ReLU? It introduces nonlinearity; two linear layers without a nonlinear activation collapse to a single linear transformation.
3. What is validation for? Selecting the checkpoint without optimizing against test performance.
4. Why compare logistic regression? It tests whether the extra nonlinear model improves this specific task over a simple reference.
5. Why can a 99% score be wrong? Softmax normalizes logits; it does not guarantee calibrated certainty or recognize unfamiliar inputs.
6. Why not claim 95.55% on uploaded photos? The measurements cover the prepared UCI images only.
7. How do you verify the demo? Re-run the browser forward pass on all test images and compare its outputs with PyTorch.

بالعراقي: افهم تقسيم البيانات، شلون نحسب الخطأ، وليش نختار نسخة النموذج حسب الـvalidation. جرّب تشغيل الكود بنفسك وتغيير صغير تگدر تشرح أثره. النتائج هنا انحسبت فعلًا، بس ما تعني إن النموذج يقرأ أي صورة. وضّح مساهمتك الحالية بصدق، ووسّعها كلما تجرب وتتعلم.
