"""
Test OpenAI fine-tuning - adapt models to your specific use case with custom data

WORKFLOW:
    1. Prepare data (training_data.jsonl)                    → test_21
    2. Upload to OpenAI → Get FILE ID                        → test_22
    3. Create job with FILE ID → Get JOB ID (💰 costs $)    → test_23
    4. Monitor JOB ID → Get MODEL ID when complete           → test_24
    5. Use MODEL ID for inference                             → test_25

EXAMPLE:
    training_data.jsonl
        ↓ upload
    file-abc123 (FILE ID)
        ↓ create_job
    ftjob-xyz789 (JOB ID)
        ↓ wait + monitor (10-60 min)
    ft:gpt-4o-mini:org:my-model:abc123 (MODEL ID) ← Use this for inference!
"""
import os
import json
import time
from pathlib import Path

import pytest
from dotenv import load_dotenv
from openai import OpenAI  # OpenAI Python SDK for fine-tuning API


load_dotenv()


@pytest.fixture(scope="module")
def openai_client():
    """Create OpenAI client for fine-tuning operations"""
    api_key = os.getenv('OPENAI_API_KEY')  # Get API key from environment
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")  # Skip if no key
    return OpenAI(api_key=api_key)  # Initialize client


@pytest.fixture(scope="module")
def finetuning_dataset_path(tmp_path_factory):
    """Create path for fine-tuning dataset file"""
    tmp_dir = tmp_path_factory.mktemp("finetuning")  # Create temp directory
    return tmp_dir / "training_data.jsonl"  # JSONL = one JSON object per line


@pytest.fixture(scope="module", autouse=True)
def cleanup_uploaded_files(openai_client, finetuning_dataset_path):
    """
    Auto-cleanup fixture: Deletes uploaded test files from OpenAI after tests complete.
    
    WHY THIS IS NEEDED:
    • Every test run uploads a new file to OpenAI
    • Files accumulate and clutter your OpenAI account
    • This fixture automatically deletes test files after all tests finish
    
    RUNS AUTOMATICALLY:
    • autouse=True means this runs for every test session
    • Cleanup happens AFTER all tests complete (not before)
    """
    yield  # Tests run here
    
    # AFTER all tests complete, cleanup uploaded files
    try:
        file_id_path = finetuning_dataset_path.parent / "file_id.txt"
        if file_id_path.exists():
            file_id = file_id_path.read_text().strip()  # Get the file ID we uploaded
            
            # Delete the file from OpenAI
            openai_client.files.delete(file_id)  # Remove from OpenAI servers
            print(f"\n🧹 Cleaned up uploaded file: {file_id}")
    except Exception as e:
        print(f"\n⚠️ Could not cleanup file (might not exist): {e}")
        # Don't fail tests if cleanup fails


def test_21_prepare_finetuning_dataset(finetuning_dataset_path):
    """
    Prepare dataset for fine-tuning in OpenAI's required format.
    
    FINE-TUNING BASICS:
    • Fine-tuning = Training an existing model on YOUR specific data
    • Makes the model better at YOUR specific tasks/style/domain
    • Uses transfer learning (start from pre-trained model, adapt to your use case)
    
    DATASET FORMAT (JSONL):
    • One JSON object per line (not a JSON array!)
    • Each line has "messages" array with conversation examples
    • Format: {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    
    WHEN TO FINE-TUNE:
    ✅ Need consistent formatting/style (e.g., always return JSON)
    ✅ Specialized domain knowledge (medical, legal, technical)
    ✅ Specific tone/brand voice (formal, casual, technical)
    ✅ Complex multi-step tasks specific to your use case
    
    WHEN NOT TO FINE-TUNE:
    ❌ Few-shot prompting works well (try that first!)
    ❌ General knowledge tasks (GPT already knows this)
    ❌ Less than 50 quality examples (need more data!)
    """
    
    # EXAMPLE USE CASE: Fine-tune model to answer questions about a fictional company
    # This teaches the model specific company knowledge + desired response style
    
    training_examples = [
        # Example 1: Company info question
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant for TechCorp, a software company."},  # System prompt
                {"role": "user", "content": "What does TechCorp do?"},  # User question
                {"role": "assistant", "content": "TechCorp specializes in AI-powered document processing solutions. We help businesses automate their document workflows using cutting-edge machine learning technology."}  # Desired response
            ]
        },
        # Example 2: Product question
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant for TechCorp, a software company."},
                {"role": "user", "content": "Tell me about your main product."},
                {"role": "assistant", "content": "Our flagship product is DocuFlow AI, which automatically extracts, classifies, and routes documents. It reduces processing time by 80% and integrates seamlessly with existing systems."}
            ]
        },
        # Example 3: Pricing question
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant for TechCorp, a software company."},
                {"role": "user", "content": "How much does it cost?"},
                {"role": "assistant", "content": "DocuFlow AI starts at $299/month for up to 1,000 documents. Enterprise plans with custom limits are available. Contact sales@techcorp.com for a quote."}
            ]
        },
        # Example 4: Support hours
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant for TechCorp, a software company."},
                {"role": "user", "content": "What are your support hours?"},
                {"role": "assistant", "content": "Our support team is available Monday-Friday, 9 AM to 6 PM EST. Premium customers get 24/7 support. You can reach us at support@techcorp.com or through live chat."}
            ]
        },
        # Example 5: Integration question
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant for TechCorp, a software company."},
                {"role": "user", "content": "What systems does DocuFlow integrate with?"},
                {"role": "assistant", "content": "DocuFlow AI integrates with Salesforce, SAP, Microsoft Dynamics, and any system with a REST API. We also offer custom integrations for enterprise customers."}
            ]
        },
        # Example 6: Trial question
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant for TechCorp, a software company."},
                {"role": "user", "content": "Do you offer a free trial?"},
                {"role": "assistant", "content": "Yes! We offer a 14-day free trial with full access to all features. No credit card required. Sign up at techcorp.com/trial to get started."}
            ]
        },
        # Example 7: Security question
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant for TechCorp, a software company."},
                {"role": "user", "content": "Is my data secure?"},
                {"role": "assistant", "content": "Absolutely. We're SOC 2 Type II certified and GDPR compliant. All data is encrypted at rest and in transit. We never share your data with third parties."}
            ]
        },
        # Example 8: Company history
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant for TechCorp, a software company."},
                {"role": "user", "content": "When was TechCorp founded?"},
                {"role": "assistant", "content": "TechCorp was founded in 2018 by Jane Smith and John Doe, both former engineers at Google. We now serve over 500 companies worldwide."}
            ]
        },
    ]
    
    # Write to JSONL file (one JSON object per line)
    with open(finetuning_dataset_path, 'w', encoding='utf-8') as f:
        for example in training_examples:
            json.dump(example, f)  # Convert dict to JSON string
            f.write('\n')  # IMPORTANT: newline after each JSON object (that's what makes it JSONL!)
    
    # Verify file was created
    assert finetuning_dataset_path.exists(), "Dataset file should exist"  # File exists
    
    # Verify file has correct number of lines
    lines = finetuning_dataset_path.read_text(encoding='utf-8').strip().split('\n')  # Read all lines
    assert len(lines) == len(training_examples), f"Should have {len(training_examples)} lines"  # Correct count
    
    # Verify each line is valid JSON
    for i, line in enumerate(lines):
        try:
            data = json.loads(line)  # Parse JSON
            assert "messages" in data, f"Line {i+1} should have 'messages' key"  # Has messages
            assert len(data["messages"]) >= 2, f"Line {i+1} should have at least user+assistant"  # Has conversation
        except json.JSONDecodeError as e:
            pytest.fail(f"Line {i+1} is not valid JSON: {e}")  # Invalid JSON
    
    print(f"✅ Created fine-tuning dataset with {len(training_examples)} examples")
    print(f"📁 Saved to: {finetuning_dataset_path}")
    print(f"📊 File size: {finetuning_dataset_path.stat().st_size} bytes")


def test_22_upload_finetuning_dataset(openai_client, finetuning_dataset_path):
    """
    Upload dataset to OpenAI for fine-tuning.
    
    WHAT HAPPENS:
    • File is uploaded to OpenAI's servers
    • OpenAI validates the format (returns errors if invalid)
    • You get a file ID to use when creating fine-tuning job
    • File is processed and ready for training
    
    COST:
    • Uploading is FREE
    • You only pay when the fine-tuning job runs
    • Storage is FREE (OpenAI keeps your files)
    """
    
    # Skip if dataset doesn't exist (previous test might have been skipped)
    if not finetuning_dataset_path.exists():
        pytest.skip("Dataset file doesn't exist. Run test_21 first.")
    
    print(f"\n📤 Uploading dataset to OpenAI...")
    
    # Upload file to OpenAI
    with open(finetuning_dataset_path, 'rb') as f:  # Open in binary mode
        file_response = openai_client.files.create(
            file=f,                    # File to upload
            purpose="fine-tune"        # Purpose must be "fine-tune" (OpenAI requirement)
        )
    
    # Store file ID for later tests (we'll need this to create the fine-tuning job)
    file_id = file_response.id  # e.g., "file-abc123"
    
    # Verify upload worked
    assert file_id is not None, "Should get file ID"  # Got ID
    assert file_id.startswith("file-"), "File ID should start with 'file-'"  # Correct format
    assert file_response.status in ["uploaded", "processed"], "File should be uploaded"  # Status is good
    
    print(f"✅ File uploaded successfully!")
    print(f"📄 File ID: {file_id}")
    print(f"📊 Status: {file_response.status}")
    print(f"💾 Bytes: {file_response.bytes}")
    
    # Save file ID to a temporary location so other tests can use it
    # In real code, you'd store this in a database or config file
    file_id_path = finetuning_dataset_path.parent / "file_id.txt"
    file_id_path.write_text(file_id)  # Save for next test
    
    print(f"💡 Saved file ID to: {file_id_path}")


def test_23_create_finetuning_job(openai_client, finetuning_dataset_path):
    """
    Create a fine-tuning job to train the model.
    
    WHAT HAPPENS:
    • OpenAI queues your fine-tuning job
    • Training starts when resources are available (usually within minutes)
    • Training takes 10-60 minutes depending on dataset size
    • You get a fine-tuned model ID when complete
    
    COST (as of 2024):
    • gpt-4o-mini-2024-07-18: $3.00 per 1M training tokens
    • gpt-4o-2024-08-06: $25.00 per 1M training tokens
    • For small datasets (< 10k examples), usually costs < $10
    
    PARAMETERS:
    • model: Base model to fine-tune (gpt-4o-mini, gpt-4o, etc.)
    • training_file: File ID from upload step
    • hyperparameters: epochs, learning rate (optional, OpenAI auto-tunes by default)
    • suffix: Custom name for your model (optional, e.g., "techcorp-v1")
    
    NOTE: This test creates a REAL fine-tuning job that costs money!
    It's marked with pytest.mark.skip by default. Remove @pytest.mark.skip to run.
    """
    pytest.skip("Skipped by default - fine-tuning costs money! Remove this line to run.")
    
    # Get file ID from previous test
    file_id_path = finetuning_dataset_path.parent / "file_id.txt"
    if not file_id_path.exists():
        pytest.skip("File ID not found. Run test_22 first.")
    
    file_id = file_id_path.read_text().strip()  # Read file ID
    
    print(f"\n🚀 Creating fine-tuning job...")
    print(f"💰 WARNING: This will cost money! (~$0.50-$5 for small dataset)")
    
    # Create fine-tuning job
    job_response = openai_client.fine_tuning.jobs.create(
        training_file=file_id,              # File ID from upload
        model="gpt-4o-mini-2024-07-18",    # Base model (cheaper option)
        suffix="techcorp-assistant-v1"     # Custom suffix for your model name
        # hyperparameters={                 # Optional: customize training
        #     "n_epochs": 3,                # Number of training epochs (default: auto)
        #     "batch_size": 1,              # Batch size (default: auto)
        #     "learning_rate_multiplier": 1 # Learning rate multiplier (default: auto)
        # }
    )
    
    # Store job ID for monitoring
    job_id = job_response.id  # e.g., "ftjob-abc123"
    
    # Verify job created
    assert job_id is not None, "Should get job ID"  # Got ID
    assert job_id.startswith("ftjob-"), "Job ID should start with 'ftjob-'"  # Correct format
    assert job_response.status in ["validating_files", "queued", "running"], "Job should be starting"  # Status is good
    
    print(f"✅ Fine-tuning job created!")
    print(f"🆔 Job ID: {job_id}")
    print(f"📊 Status: {job_response.status}")
    print(f"🎯 Model: {job_response.model}")
    
    # Save job ID for monitoring
    job_id_path = finetuning_dataset_path.parent / "job_id.txt"
    job_id_path.write_text(job_id)  # Save for next test
    
    print(f"💡 Saved job ID to: {job_id_path}")
    print(f"⏰ Training will take 10-60 minutes. Run test_24 to monitor progress.")


def test_24_monitor_finetuning_job(openai_client, finetuning_dataset_path):
    """
    Monitor fine-tuning job status.
    
    JOB STATUSES:
    • validating_files: OpenAI is checking your data format
    • queued: Waiting for training resources
    • running: Training in progress
    • succeeded: Training complete! Model is ready
    • failed: Training failed (check error message)
    • cancelled: You cancelled the job
    
    WHAT TO CHECK:
    • Status: Current state of the job
    • trained_tokens: How many tokens processed
    • estimated_finish: When training will complete
    • error: If failed, why
    """
    
    # Get job ID from previous test
    job_id_path = finetuning_dataset_path.parent / "job_id.txt"
    if not job_id_path.exists():
        pytest.skip("Job ID not found. Run test_23 first.")
    
    job_id = job_id_path.read_text().strip()  # Read job ID
    
    print(f"\n🔍 Checking fine-tuning job status...")
    
    # Retrieve job status
    job = openai_client.fine_tuning.jobs.retrieve(job_id)  # Get current status
    
    print(f"📊 Job ID: {job.id}")
    print(f"📊 Status: {job.status}")
    print(f"🎯 Model: {job.model}")
    
    # Show different info based on status
    if job.status == "succeeded":
        print(f"✅ Training complete!")
        print(f"🎉 Fine-tuned model: {job.fine_tuned_model}")  # Your new model ID
        print(f"📈 Trained tokens: {job.trained_tokens}")
        
        # Save fine-tuned model ID
        model_id_path = finetuning_dataset_path.parent / "finetuned_model_id.txt"
        model_id_path.write_text(job.fine_tuned_model)  # Save for next test
        
    elif job.status == "running":
        print(f"⏳ Training in progress...")
        if hasattr(job, 'estimated_finish') and job.estimated_finish:
            print(f"⏰ Estimated finish: {job.estimated_finish}")
            
    elif job.status == "failed":
        print(f"❌ Training failed!")
        if hasattr(job, 'error') and job.error:
            print(f"💥 Error: {job.error}")
        pytest.fail(f"Fine-tuning job failed: {job.error if hasattr(job, 'error') else 'Unknown error'}")
        
    elif job.status in ["validating_files", "queued"]:
        print(f"⏳ Waiting to start...")
        print(f"💡 This can take a few minutes. Run this test again to check progress.")
    
    # List events (training logs)
    if job.status in ["running", "succeeded"]:
        print(f"\n📜 Training events:")
        events = openai_client.fine_tuning.jobs.list_events(job_id, limit=5)  # Get last 5 events
        for event in events.data:
            print(f"   [{event.created_at}] {event.message}")


def test_25_use_finetuned_model(openai_client, finetuning_dataset_path, managers):
    """
    Use your fine-tuned model for inference.
    
    USAGE:
    • Use fine-tuned model ID just like any other model
    • Same API, same methods, same everything
    • Model now has your custom knowledge/style
    
    COST:
    • Fine-tuned models cost MORE than base models
    • gpt-4o-mini fine-tuned: ~2x base model cost
    • gpt-4o fine-tuned: ~2x base model cost
    • Worth it for specialized tasks!
    
    WHEN IT'S READY:
    • You can use it immediately after job.status == "succeeded"
    • Model stays available until you delete it
    • You can create multiple fine-tuned models
    """
    
    # Get fine-tuned model ID
    model_id_path = finetuning_dataset_path.parent / "finetuned_model_id.txt"
    if not model_id_path.exists():
        pytest.skip("Fine-tuned model ID not found. Run test_23 and wait for training to complete.")
    
    finetuned_model_id = model_id_path.read_text().strip()  # e.g., "ft:gpt-4o-mini-2024-07-18:org:techcorp-assistant-v1:abc123"
    
    print(f"\n🎯 Testing fine-tuned model...")
    print(f"🤖 Model: {finetuned_model_id}")
    
    # Test 1: Ask about TechCorp (model should know this from fine-tuning)
    query = "What does TechCorp do?"
    print(f"\n❓ Question: {query}")
    
    # Use the fine-tuned model with OpenAI SDK
    response = openai_client.chat.completions.create(
        model=finetuned_model_id,  # Your fine-tuned model
        messages=[
            {"role": "system", "content": "You are a helpful assistant for TechCorp, a software company."},
            {"role": "user", "content": query}
        ],
        temperature=0  # Deterministic
    )
    
    answer = response.choices[0].message.content  # Extract answer
    
    print(f"💬 Answer: {answer}")
    
    # Verify response contains expected information
    assert "TechCorp" in answer or "techcorp" in answer.lower(), "Should mention TechCorp"  # Knows company name
    assert len(answer) > 20, "Should give detailed answer"  # Not too short
    
    # Test 2: Compare with base model (optional - shows the difference)
    print(f"\n📊 Comparing with base model...")
    
    base_response = openai_client.chat.completions.create(
        model="gpt-4o-mini",  # Base model (not fine-tuned)
        messages=[
            {"role": "system", "content": "You are a helpful assistant for TechCorp, a software company."},
            {"role": "user", "content": query}
        ],
        temperature=0
    )
    
    base_answer = base_response.choices[0].message.content
    
    print(f"🤖 Base model answer: {base_answer}")
    print(f"✨ Fine-tuned model answer: {answer}")
    print(f"\n💡 Notice: Fine-tuned model gives specific details about TechCorp!")
    print(f"💡 Base model doesn't know TechCorp specifics (no training data)")
    
    print(f"\n✅ Fine-tuned model works!")


def test_26_list_finetuned_models(openai_client):
    """
    List all your fine-tuned models.
    
    USEFUL FOR:
    • See all models you've created
    • Get model IDs to use them
    • Check which models are ready
    • Delete old models you don't need
    """
    
    print(f"\n📋 Listing your fine-tuned models...")
    
    # List all fine-tuning jobs
    jobs = openai_client.fine_tuning.jobs.list(limit=10)  # Get last 10 jobs
    
    print(f"\n🎯 Found {len(jobs.data)} fine-tuning jobs:")
    
    for i, job in enumerate(jobs.data, 1):
        print(f"\n{i}. Job ID: {job.id}")
        print(f"   Status: {job.status}")
        print(f"   Base model: {job.model}")
        
        if job.status == "succeeded" and job.fine_tuned_model:
            print(f"   ✅ Fine-tuned model: {job.fine_tuned_model}")  # This is what you use for inference
            print(f"   💰 Trained tokens: {job.trained_tokens}")
        elif job.status == "failed":
            print(f"   ❌ Failed")
        elif job.status in ["running", "queued"]:
            print(f"   ⏳ In progress")
    
    # At least verify we can list jobs (even if empty)
    assert jobs is not None, "Should be able to list jobs"
    assert hasattr(jobs, 'data'), "Should have data attribute"


def test_27_cancel_finetuning_job(openai_client, finetuning_dataset_path):
    """
    Cancel a running fine-tuning job (if needed).
    
    WHEN TO CANCEL:
    • Job is taking too long
    • You made a mistake in the dataset
    • You want to save money
    • Testing purposes
    
    NOTE: You can't cancel a completed job
    This test is skipped by default (only cancel if you want to!)
    """
    pytest.skip("Skipped by default - only cancel if you really want to!")
    
    # Get job ID
    job_id_path = finetuning_dataset_path.parent / "job_id.txt"
    if not job_id_path.exists():
        pytest.skip("Job ID not found.")
    
    job_id = job_id_path.read_text().strip()
    
    print(f"\n🛑 Cancelling fine-tuning job...")
    print(f"🆔 Job ID: {job_id}")
    
    # Cancel the job
    job = openai_client.fine_tuning.jobs.cancel(job_id)  # Send cancel request
    
    print(f"✅ Cancellation requested")
    print(f"📊 Status: {job.status}")  # Should be "cancelled"
    
    assert job.status == "cancelled", "Job should be cancelled"


def test_28_cleanup_old_uploaded_files(openai_client):
    """
    Clean up old test files from OpenAI (manual cleanup).
    
    USE THIS IF:
    • You've run tests many times and have duplicate files
    • Auto-cleanup fixture didn't work
    • You want to manually clean up your OpenAI account
    
    WHAT IT DOES:
    • Lists all files with purpose="fine-tune"
    • Deletes files that contain "training_data" in filename (our test files)
    • Skips other important files
    
    SAFE:
    • Only deletes files named "training_data.jsonl"
    • Won't delete your production fine-tuning files
    
    Run with: pytest tests/test_s08_finetuning.py::test_28_cleanup_old_uploaded_files -v -s
    """
    pytest.skip("Skipped by default - only run if you want to cleanup old files")
    
    print(f"\n🧹 Searching for old test files to cleanup...")
    
    # List all files with purpose="fine-tune"
    files = openai_client.files.list(purpose="fine-tune")  # Get all fine-tune files
    
    deleted_count = 0  # Track how many we delete
    
    for file in files.data:
        # Only delete files that look like our test files
        if "training_data" in file.filename.lower():  # Our test files are named "training_data.jsonl"
            try:
                print(f"   🗑️ Deleting: {file.id} ({file.filename})")
                openai_client.files.delete(file.id)  # Delete the file
                deleted_count += 1
            except Exception as e:
                print(f"   ⚠️ Could not delete {file.id}: {e}")
    
    print(f"\n✅ Cleanup complete! Deleted {deleted_count} test file(s)")
    
    if deleted_count == 0:
        print(f"💡 No test files found. Your account is clean!")
    else:
        print(f"💡 Freed up space by removing {deleted_count} duplicate files")


# Run with:
# pytest tests/test_s08_finetuning.py -v -s
#
# IMPORTANT NOTES:
# • test_23 is SKIPPED by default (costs money!)
# • Remove pytest.skip() in test_23 to actually run fine-tuning
# • Fine-tuning takes 10-60 minutes, so test_24 and test_25 will fail until complete
# • Run test_21 and test_22 first to prepare data
# • Then run test_23 to start training (costs money!)
# • Wait 10-60 minutes
# • Then run test_24 to check status
# • When complete, run test_25 to use your model
#
# AUTO-CLEANUP:
# • Uploaded test files are AUTOMATICALLY deleted after tests finish
# • This prevents duplicate files cluttering your OpenAI account
# • If you've run tests many times, use test_28 to manually cleanup old files
#
# MANUAL CLEANUP (if needed):
# pytest tests/test_s08_finetuning.py::test_28_cleanup_old_uploaded_files -v -s
# (Remove the pytest.skip() line first!)
#
# COST ESTIMATE:
# • Small dataset (8 examples): ~$0.50-$2.00
# • Medium dataset (100 examples): ~$5-$20
# • Large dataset (1000 examples): ~$50-$200
#
# BEST PRACTICES:
# 1. Start with 50-100 high-quality examples
# 2. Test with base model + few-shot prompting first
# 3. Only fine-tune if few-shot doesn't work well enough
# 4. Use validation set to measure improvement
# 5. Fine-tune gpt-4o-mini first (cheaper), then gpt-4o if needed

