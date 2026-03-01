"""
RentShield AI Analysis Engine - Demo Script

Demonstrates all API endpoints with sample data.
Requires Ollama with Mistral model to be running.

Usage:
    python test_demo.py
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from PIL import Image

# Configuration
BASE_URL = "http://localhost:8000"
TIMEOUT = 120


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_json(data: dict) -> None:
    """Print formatted JSON."""
    print(json.dumps(data, indent=2, default=str))


def create_test_image(path: Path) -> None:
    """Create a test image for evidence validation."""
    img = Image.new("RGB", (800, 600), color=(200, 100, 100))
    img.save(str(path), "JPEG")
    print(f"  Created test image: {path}")


def test_health() -> bool:
    """Test the health endpoint."""
    print_header("1. Health Check")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        data = response.json()
        print_json(data)
        
        if data.get("llm_connected"):
            print("\n  ✅ LLM connection successful!")
            return True
        else:
            print("\n  ⚠️ LLM not connected - some tests may fail")
            return False
            
    except requests.exceptions.ConnectionError:
        print("  ❌ Cannot connect to API. Is the server running?")
        print("  Run: uvicorn app.main:app --reload --port 8000")
        return False


def test_classify_issues() -> None:
    """Test issue classification with sample issues."""
    print_header("2. Classify Issues")
    
    sample_issues = [
        {
            "description": "Water has been leaking from the ceiling for 2 weeks causing mold growth. I've reported this issue multiple times but no repairs have been made. The mold is affecting my family's health.",
            "evidence_count": 3,
        },
        {
            "description": "The landlord entered my apartment without any notice while I was at work. I have a security camera that recorded the entry. This is the third time this has happened.",
            "evidence_count": 1,
        },
        {
            "description": "The heating system has been completely broken for 3 weeks during winter. Indoor temperature drops to 50°F at night. I have elderly parents living with me and this is a serious health concern.",
            "evidence_count": 2,
        },
    ]
    
    for i, issue in enumerate(sample_issues, 1):
        print(f"\n  Issue #{i}:")
        print(f"  Description: {issue['description'][:80]}...")
        
        try:
            start = time.time()
            response = requests.post(
                f"{BASE_URL}/api/v1/classify-issue",
                json=issue,
                timeout=TIMEOUT,
            )
            elapsed = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                print(f"  Category: {data['primary_category']}")
                print(f"  Severity: {data['severity']}")
                print(f"  Confidence: {data['confidence']}%")
                print(f"  Urgency: {'🚨 YES' if data['urgency_flag'] else 'No'}")
                print(f"  Time: {elapsed:.2f}s")
            else:
                print(f"  ❌ Error: {response.status_code}")
                print_json(response.json())
                
        except Exception as e:
            print(f"  ❌ Error: {e}")


def test_validate_evidence(test_data_dir: Path) -> None:
    """Test evidence validation with sample images."""
    print_header("3. Validate Evidence")
    
    # Create test image
    test_image = test_data_dir / "test_evidence.jpg"
    create_test_image(test_image)
    
    try:
        print("\n  Validating test image...")
        start = time.time()
        
        with open(test_image, "rb") as f:
            response = requests.post(
                f"{BASE_URL}/api/v1/validate-evidence",
                files={"image": ("evidence.jpg", f, "image/jpeg")},
                data={
                    "claim_text": "Water damage visible on kitchen ceiling",
                    "incident_date": "2024-01-15T00:00:00Z",
                },
                timeout=TIMEOUT,
            )
        
        elapsed = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            print(f"  Authenticity Score: {data['authenticity_score']}/100")
            print(f"  Trust Level: {data['trust_level']}")
            print(f"  Tamper Probability: {data['tamper_analysis']['tamper_probability']:.0%}")
            print(f"  EXIF Dimensions: {data['exif_data'].get('dimensions', 'N/A')}")
            if data.get("alignment_analysis"):
                print(f"  Alignment Score: {data['alignment_analysis']['alignment_score']}/100")
            print(f"  Time: {elapsed:.2f}s")
        else:
            print(f"  ❌ Error: {response.status_code}")
            print_json(response.json())
            
    except Exception as e:
        print(f"  ❌ Error: {e}")


def test_analyze_case() -> None:
    """Test full case analysis."""
    print_header("4. Analyze Full Case")
    
    case_data = {
        "issue_id": "demo-case-001",
        "tenant_complaint": """
            My apartment has had a severe water leak from the upstairs unit for the past 3 weeks.
            I first reported this issue on January 10th via email and received no response.
            I followed up with phone calls on January 12th and 15th.
            The landlord sent a maintenance person on January 18th who said the repair was "too extensive"
            and they would need to call a plumber. No plumber ever came.
            
            The leak has caused visible mold growth on the ceiling and walls. I'm concerned about
            health effects - my son has developed respiratory symptoms since this started.
            
            I have photos of the damage taken on multiple dates, emails showing my repair requests,
            and a receipt from the doctor's visit for my son.
        """,
        "landlord_response": """
            We dispatched our maintenance team promptly after receiving the complaint.
            Our technician assessed the damage and determined a specialized plumber was needed.
            We have been working to schedule a plumber but have faced delays due to
            contractor availability. The tenant has also been difficult to reach to confirm
            appointment times. We are committed to resolving this issue.
        """,
        "incident_date": "2024-01-10T00:00:00Z",
        "tenant_evidence": [
            {
                "file_url": "https://example.com/evidence/leak_photo_1.jpg",
                "uploaded_at": "2024-01-12T10:00:00Z",
            },
            {
                "file_url": "https://example.com/evidence/mold_photo.jpg",
                "uploaded_at": "2024-01-20T15:00:00Z",
            },
        ],
        "landlord_evidence": [
            {
                "file_url": "https://example.com/evidence/work_order.pdf",
                "uploaded_at": "2024-01-25T09:00:00Z",
            },
        ],
        "property_history": {
            "previous_complaints": 8,
            "resolution_rate": 0.5,
        },
    }
    
    try:
        print("\n  Submitting case for analysis...")
        print("  (This may take 1-2 minutes for comprehensive AI analysis)")
        
        start = time.time()
        response = requests.post(
            f"{BASE_URL}/api/v1/analyze-case",
            json=case_data,
            timeout=180,  # Extended timeout for full analysis
        )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"\n  📋 Case Summary:")
            print(f"     {data['case_summary']}")
            
            print(f"\n  👤 Tenant Position:")
            print(f"     Evidence Strength: {data['tenant_position']['evidence_strength']}/100")
            print(f"     Credibility: {data['tenant_position']['credibility_score']}/100")
            
            print(f"\n  🏠 Landlord Position:")
            print(f"     Evidence Strength: {data['landlord_position']['evidence_strength']}/100")
            print(f"     Credibility: {data['landlord_position']['credibility_score']}/100")
            
            print(f"\n  🗳️ DAO Recommendation:")
            rec = data["dao_recommendation"]
            print(f"     Outcome: {rec['recommended_outcome']}")
            print(f"     Confidence: {rec['confidence_level'].upper()}")
            print(f"     Tenant Favor: {rec['tenant_favor_confidence']}%")
            print(f"     Landlord Favor: {rec['landlord_favor_confidence']}%")
            print(f"     Suggested Resolution: {rec['suggested_resolution']}")
            
            print(f"\n  ⚠️ Red Flags Detected:")
            red_flags = data["red_flags"]
            if red_flags["tenant_concerns"]:
                print(f"     Tenant: {', '.join(red_flags['tenant_concerns'][:2])}")
            if red_flags["landlord_concerns"]:
                print(f"     Landlord: {', '.join(red_flags['landlord_concerns'][:2])}")
            if not red_flags["tenant_concerns"] and not red_flags["landlord_concerns"]:
                print("     None detected")
            
            print(f"\n  📅 Estimated Timeline: {data['estimated_resolution_timeline']}")
            print(f"\n  ⏱️ Analysis Time: {elapsed:.2f}s")
            
        else:
            print(f"  ❌ Error: {response.status_code}")
            print_json(response.json())
            
    except Exception as e:
        print(f"  ❌ Error: {e}")


def main() -> None:
    """Run all demo tests."""
    print("\n" + "=" * 60)
    print("  🏠 RentShield AI Analysis Engine - Demo")
    print("=" * 60)
    print(f"  API URL: {BASE_URL}")
    print(f"  Time: {datetime.now().isoformat()}")
    
    start_time = time.time()
    
    # Create test data directory
    test_data_dir = Path("test_data")
    test_data_dir.mkdir(exist_ok=True)
    (test_data_dir / ".gitkeep").touch()
    
    # Run tests
    if not test_health():
        print("\n❌ Health check failed. Exiting.")
        sys.exit(1)
    
    test_classify_issues()
    test_validate_evidence(test_data_dir)
    test_analyze_case()
    
    total_time = time.time() - start_time
    
    print_header("Demo Complete")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  All tests completed! ✅")
    print()


if __name__ == "__main__":
    main()
