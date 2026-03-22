FROM python:3.12-slim

# Run as non-root for security
RUN addgroup --system app && adduser --system --ingroup app app

WORKDIR /app

# Install deps first — layer cached unless requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

USER app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--log-level", "info"]
