#!/usr/bin/env python
"""Fetch real IT jobs from MyFutureJobs API (Malaysia)."""

import json
import time
import re
from pathlib import Path

import requests

# Industry codes for IT & Communications (MyFutureJobs taxonomy)
INDUSTRY_CODES = [
    "6201",  # Computer programming activities
    "6202",  # Computer consultancy activities
    "6203",  # Computer facilities management
    "6209",  # Other information technology service activities
    "6101",  # Wired telecommunications
    "6102",  # Wireless telecommunications
    "6110",  # Satellite communications
]

# Location codes: Kuala Lumpur (MY10), Selangor (MY12)
LOCATION_CODES = ["MY10", "MY12"]

def parse_salary(salary_str):
    """Extract numeric min and max from salary string like 'RM 5000 - 7000' or 'RM 6000'."""
    if not salary_str or salary_str == "Not specified":
        return 0, 0, salary_str
    # Remove 'RM' and commas
    cleaned = re.sub(r'[RM,]+', '', salary_str).strip()
    numbers = re.findall(r'\d+', cleaned)
    if not numbers:
        return 0, 0, salary_str
    if len(numbers) == 1:
        val = int(numbers[0])
        return val, val, salary_str
    else:
        return int(numbers[0]), int(numbers[1]), salary_str

def fetch_jobs_page(page=1, per_page=50):
    url = "https://www.myfuturejobs.gov.my/api/v1/jobs/search"
    params = {
        "page": page,
        "per_page": per_page,
        "industry[]": INDUSTRY_CODES,
        "location[]": LOCATION_CODES,
        "sort": "posted_date_desc",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()

def main():
    all_jobs = []
    page = 1
    print("Fetching jobs from MyFutureJobs API...")
    print(f"Industries: {INDUSTRY_CODES}")
    print(f"Locations: {LOCATION_CODES}")
    while True:
        print(f"  Page {page}...")
        try:
            data = fetch_jobs_page(page)
        except requests.exceptions.RequestException as e:
            print(f"  Error on page {page}: {e}")
            break
        except Exception as e:
            print(f"  Unexpected error: {e}")
            break
        
        jobs = data.get("data", [])
        if not jobs:
            print("  No more jobs found.")
            break
        
        for job in jobs:
            title = job.get("title", "").strip()
            company = job.get("company_name", "").strip()
            description = job.get("description", "").strip()
            salary_raw = job.get("salary", "Not specified")
            salary_min, salary_max, salary_display = parse_salary(salary_raw)
            
            all_jobs.append({
                "title": title,
                "company": company,
                "description": description,
                "salary_display": salary_display,
                "salary_min": salary_min,
                "salary_max": salary_max,
            })
        
        pagination = data.get("pagination", {})
        if page >= pagination.get("last_page", page):
            break
        page += 1
        time.sleep(0.5)  # be polite to the API
    
    # Save to data folder
    output_path = Path(__file__).parent.parent / "data" / "jobs_snapshot.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_jobs, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved {len(all_jobs)} real job listings to {output_path}")
    if all_jobs:
        print("\nSample job:")
        sample = all_jobs[0]
        print(f"  Title: {sample['title']}")
        print(f"  Company: {sample['company']}")
        print(f"  Salary: {sample['salary_display']} (min {sample['salary_min']}, max {sample['salary_max']})")
        print(f"  Description preview: {sample['description'][:150]}...")
    else:
        print("\n⚠️ No jobs fetched. The API may be down or returned empty.")
        print("   You can continue using the sample jobs in data/jobs_snapshot.json.")

if __name__ == "__main__":
    main()