# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import pandas as pd

# # -----------------------------
# # Sample Resume Data
# # -----------------------------
# resumes = [
#     "Python Java SQL Machine Learning Data Science",
#     "HTML CSS JavaScript React Node MongoDB",
#     "C C++ Java Operating Systems Computer Networks"
# ]

# resume_ids = [101, 102, 103]

# # -----------------------------
# # Sample Job Description Data
# # -----------------------------
# jobs = [
#     "Looking for Python developer with machine learning and data analysis skills",
#     "Frontend developer required with React and JavaScript experience",
#     "System engineer with operating systems and networking knowledge"
# ]

# job_ids = [201, 202, 203]

# # -----------------------------
# # Combine all text
# # -----------------------------
# documents = resumes + jobs

# # -----------------------------
# # TF-IDF Vectorization
# # -----------------------------
# vectorizer = TfidfVectorizer(stop_words='english')
# tfidf_matrix = vectorizer.fit_transform(documents)

# # -----------------------------
# # Split resume and job vectors
# # -----------------------------
# resume_vectors = tfidf_matrix[:len(resumes)]
# job_vectors = tfidf_matrix[len(resumes):]

# # -----------------------------
# # Cosine Similarity Calculation
# # -----------------------------
# similarity_matrix = cosine_similarity(resume_vectors, job_vectors)

# # -----------------------------
# # Prepare Recommendation Result
# # -----------------------------
# results = []

# for i, resume_id in enumerate(resume_ids):
#     for j, job_id in enumerate(job_ids):
#         results.append({
#             "resume_id": resume_id,
#             "job_id": job_id,
#             "similarity_score": round(similarity_matrix[i][j], 3)
#         })

# # -----------------------------
# # Display Result
# # -----------------------------
# df = pd.DataFrame(results)
# print("\nJob Recommendation Scores:\n")
# print(df.sort_values(by="similarity_score", ascending=False))

# # -----------------------------
# # Recommendation Function
# # -----------------------------
# def recommend_jobs_for_resume(resume_index, top_n=3):
#     similarities = cosine_similarity(
#         resume_vectors[resume_index], job_vectors
#     )[0]

#     ranked_jobs = sorted(
#         enumerate(similarities),
#         key=lambda x: x[1],
#         reverse=True
#     )

#     return ranked_jobs[:top_n]


# # -----------------------------
# # Main Execution
# # -----------------------------
# if __name__ == "__main__":
#     print("AI Job Recommendation Results\n")

#     for i, resume_id in enumerate(resume_ids):
#         print(f"Resume ID: {resume_id}")
#         recommendations = recommend_jobs_for_resume(i)

#         for job_index, score in recommendations:
#             print(
#                 f"  Job ID: {job_ids[job_index]} | Similarity Score: {round(score, 2)}"
#             )
#         print("-" * 40)


# """We implemented a content-based AI job recommendation system using TF-IDF vectorization 
# and cosine similarity. Resumes and job descriptions are converted into numerical vectors,
#  and similarity scores are computed to recommend the most relevant jobs for each candidate."""



from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Initialize Flask App
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Sample Resume & Job Data
# -----------------------------
resumes = [
    "Python Java SQL Machine Learning Data Science",
    "HTML CSS JavaScript React Node MongoDB",
    "C C++ Java Operating Systems Computer Networks"
]

resume_ids = [101, 102, 103]

jobs = [
    "Looking for Python developer with machine learning and data analysis skills",
    "Frontend developer required with React and JavaScript experience",
    "System engineer with operating systems and networking knowledge"
]

job_ids = [201, 202, 203]

# -----------------------------
# TF-IDF Vectorization
# -----------------------------
documents = resumes + jobs

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)

resume_vectors = tfidf_matrix[:len(resumes)]
job_vectors = tfidf_matrix[len(resumes):]

# -----------------------------
# Recommendation Logic
# -----------------------------
def recommend_jobs(resume_text, top_n=3):
    resume_vector = vectorizer.transform([resume_text])
    similarity_scores = cosine_similarity(resume_vector, job_vectors)[0]

    ranked_jobs = sorted(
        enumerate(similarity_scores),
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []
    for index, score in ranked_jobs[:top_n]:
        recommendations.append({
            "job_id": job_ids[index],
            "job_description": jobs[index],
            "similarity_score": round(float(score), 2)
        })

    return recommendations

# -----------------------------
# Flask API Endpoint
# -----------------------------
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()

    if not data or "resume" not in data:
        return jsonify({"error": "Resume text is required"}), 400

    resume_text = data["resume"]
    results = recommend_jobs(resume_text)

    return jsonify({
        "recommendations": results
    })

# -----------------------------
# Run Flask App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
