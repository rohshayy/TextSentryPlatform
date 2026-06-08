# 1. Start with an official, lightweight Linux image that already has Python 3.14 pre-installed.
FROM python:3.14-slim

# 2. Set the internal coordinate directory where our code will live inside this virtual machine.
WORKDIR /app

# 3. Copy our system dependency ledger from your local computer into the virtual machine.
COPY requirements.txt .

# 4. Run the installer inside the virtual machine to set up your PyTorch and FastAPI environments.
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy all your core AI assets, weights (.pth), configurations, and python scripts into the container.
COPY . .

# 6. Expose port 8000 so internet traffic can pass into our virtual container space.
EXPOSE 8000

# 7. The infinite execution loop command that boots up Uvicorn the split second the container wakes up.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]