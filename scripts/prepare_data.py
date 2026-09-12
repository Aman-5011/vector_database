import os
import sys
import uuid
import random

# Ensure the 'src' module can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.embedding_pipeline import EmbeddingPipeline

def generate_synthetic_dataset(total_records: int = 50000, seed: int = 42) -> list[dict]:
    """
    Generates a diverse, realistic, semantically grouped synthetic dataset.
    Uses seeded random sampling over rich grammatical templates and synonym banks.
    Guarantees 100% unique text strings with zero numeric suffixes.
    """
    rng = random.Random(seed)
    
    categories = {
        "Technology": {
            "subjects": [
                "Senior backend engineers", "Site reliability teams", "Cloud architects", 
                "Open-source contributors", "Cybersecurity analysts", "DevOps specialists", 
                "Database administrators", "System programmers", "Machine learning researchers", 
                "Infrastructure engineers", "Software development leads", "Platform engineers",
                "Full-stack developers", "Security operations teams", "Data pipeline engineers"
            ],
            "actions": [
                "refactored the legacy codebase for", "migrated core microservices to", 
                "benchmarked query latencies across", "deployed automated failover protocols for", 
                "audited role-based access permissions within", "profiled memory allocations inside", 
                "implemented zero-downtime rolling updates on", "optimized execution plans for", 
                "configured distributed caching policies across", "hardened network firewall rules on", 
                "debugged synchronization deadlocks inside", "scaled elastic worker pools for"
            ],
            "targets": [
                "the distributed key-value store", "the asynchronous event bus", 
                "multi-region container clusters", "the relational database read replicas", 
                "fault-tolerant message queues", "internal API gateway routers", 
                "high-throughput streaming pipelines", "centralized logging services", 
                "microservice communication meshes", "the time-series metrics warehouse"
            ],
            "contexts": [
                "to sustain peak holiday traffic spikes", "following a rigorous security audit", 
                "to maintain sub-ten-millisecond latency SLAs", "during the scheduled maintenance window", 
                "to eliminate single points of failure", "to drastically curb cloud infrastructure costs", 
                "ahead of the major production release", "to ensure high availability under load", 
                "as part of the multi-cloud resilience strategy", "to improve developer deployment velocity"
            ],
            "templates": [
                "{subject} {action} {target} {context}.",
                "To address critical bottlenecks, {subject} {action} {target}.",
                "{context}, {subject} {action} {target}.",
                "Engineers verified that {subject} {action} {target} {context}."
            ]
        },
        "Finance": {
            "subjects": [
                "Portfolio risk managers", "Quantitative analysts", "Institutional investors", 
                "Central banking economists", "Equity research specialists", "Commodity traders", 
                "Treasury oversight committees", "Asset allocation directors", "Hedge fund partners", 
                "Corporate financial controllers", "Credit rating analysts", "Private equity associates"
            ],
            "actions": [
                "liquidated volatile positions in", "hedged structural exposure to", 
                "diversified long-term holdings into", "rebalanced capital allocations across", 
                "assessed macroeconomic sensitivity toward", "monitored foreign currency fluctuations against", 
                "forecasted yield curve inversions affecting", "audited balance sheet reserves backing", 
                "structured derivative contracts around", "curbed excessive speculative leverage in"
            ],
            "targets": [
                "short-duration sovereign treasury bills", "high-yield corporate bond portfolios", 
                "emerging market equity indexes", "syndicated leveraged commercial loans", 
                "inflation-protected securities", "decentralized settlement instruments", 
                "cross-currency swap facilities", "multinational real estate investment trusts"
            ],
            "contexts": [
                "to mitigate downside systemic volatility", "amid tightening monetary policy conditions", 
                "ahead of anticipated interest rate hikes", "to safeguard institutional liquidity", 
                "following quarterly earnings announcements", "in response to rising inflationary pressures", 
                "to satisfy stringent regulatory capital requirements", "to guarantee capital preservation"
            ],
            "templates": [
                "{subject} {action} {target} {context}.",
                "In response to recent market movements, {subject} {action} {target}.",
                "{context}, {subject} {action} {target}.",
                "Internal memos confirm that {subject} {action} {target}."
            ]
        },
        "Healthcare": {
            "subjects": [
                "Clinical research oncologists", "Hospital triage physicians", "Epidemiology teams", 
                "Public health coordinators", "Pediatric specialists", "Infectious disease experts", 
                "Biomedical lab technicians", "Intensive care medical staff", "Pharmacology investigators", 
                "Surgical department heads", "Neurological researchers", "Preventive medicine consultants"
            ],
            "actions": [
                "administered targeted immunotherapy for", "isolated resistant microbial strains from", 
                "evaluated diagnostic imaging scans of", "monitored post-operative recovery stages during", 
                "published peer-reviewed findings on", "standardized triage decontamination protocols for", 
                "accelerated multi-phase human trials testing", "analyzed patient biomarker responses to"
            ],
            "targets": [
                "novel respiratory pathogen infections", "acute inflammatory cardiovascular disorders", 
                "congenital neurological syndromes", "drug-resistant bacterial cultures", 
                "hereditary metabolic abnormalities", "severe autoimmune disease flare-ups", 
                "complex localized trauma presentations", "experimental monoclonal antibody therapies"
            ],
            "contexts": [
                "under rigorous double-blind clinical controls", "to prevent secondary hospital transmission", 
                "within emergency critical care wards", "to establish revised patient treatment baselines", 
                "in accordance with medical ethics board approvals", "to drastically lower post-surgical complications", 
                "following extensive randomized sampling", "during multi-center hospital collaborative studies"
            ],
            "templates": [
                "{subject} {action} {target} {context}.",
                "During routine clinical assessments, {subject} {action} {target}.",
                "{context}, {subject} {action} {target}.",
                "Evidence demonstrates that {subject} {action} {target}."
            ]
        },
        "Sports": {
            "subjects": [
                "Varsity coaching staff", "Professional athletic trainers", "Team conditioning coordinators", 
                "Sports medicine therapists", "Offensive tactics analysts", "Elite marathon competitors", 
                "Defensive game strategists", "National tournament contenders", "Franchise scouting directors", 
                "Strength and endurance coaches"
            ],
            "actions": [
                "implemented rigorous interval drills for", "adjusted baseline defensive formations against", 
                "analyzed high-speed video replays of", "rehabilitated persistent hamstring strains before", 
                "fine-tuned tactical possession strategies for", "benchmarked cardiovascular endurance during", 
                "executed aggressive transition presses against", "adapted recovery nutrition regimens after"
            ],
            "targets": [
                "the regional championship playoff series", "fast-paced perimeter shooting schemes", 
                "grueling pre-season conditioning camps", "counter-attacking zone defenses", 
                "high-intensity endurance training blocks", "demanding back-to-back road matches", 
                "crucial fourth-quarter set-piece scenarios", "Olympic-distance qualifying trials"
            ],
            "contexts": [
                "to secure a decisive home-court advantage", "without risking muscular fatigue injuries", 
                "under severe weather conditions on game day", "to reverse a multi-game losing streak", 
                "to maintain peak aerobic physical readiness", "before thousands of traveling supporters", 
                "in anticipation of aggressive rival pressure", "to break long-standing tournament records"
            ],
            "templates": [
                "{subject} {action} {target} {context}.",
                "Prior to the finals, {subject} {action} {target}.",
                "{context}, {subject} {action} {target}.",
                "Reporters observed that {subject} {action} {target}."
            ]
        },
        "Travel": {
            "subjects": [
                "Solo backpackers", "Experienced alpine expedition guides", "Cultural heritage historians", 
                "Ecotourism conservationists", "Hospitality operations directors", "Travel photography crews", 
                "Adventure travel bloggers", "Maritime navigation officers", "Local excursion coordinators", 
                "Wilderness survival instructors"
            ],
            "actions": [
                "navigated treacherous high-altitude trails across", "documented preserved architectural ruins within", 
                "booked traditional eco-lodge accommodations near", "organized guided historical excursions through", 
                "charted remote coastal archipelago routes around", "photographed rare sunrise vistas overlooking", 
                "sampled authentic culinary specialties across", "arranged sustainable transit logistics across"
            ],
            "targets": [
                "isolated mountain farming villages", "dense tropical rainforest reserves", 
                "ancient cobblestone cathedral districts", "rugged volcanic island coastlines", 
                "scenic glacial fjord viewpoints", "protected biosphere sanctuaries", 
                "remote desert trading settlements", "historic UNESCO World Heritage sites"
            ],
            "contexts": [
                "while avoiding crowded tourist seasons", "relying strictly on local municipal transit", 
                "during unpredictable monsoon weather patterns", "to foster community-based tourism", 
                "following unmapped footpaths across the valley", "under the guidance of native park rangers", 
                "to capture authentic documentary footage", "with minimal environmental footprint"
            ],
            "templates": [
                "{subject} {action} {target} {context}.",
                "Early in the expedition, {subject} {action} {target}.",
                "{context}, {subject} {action} {target}.",
                "Travel journals record that {subject} {action} {target}."
            ]
        }
    }

    records_per_category = total_records // len(categories)
    dataset = []
    seen_texts = set()

    print(f"Generating {total_records} semantically rich sentences ({records_per_category} per category)...")

    for category_name, data in categories.items():
        generated_count = 0
        while generated_count < records_per_category:
            template = rng.choice(data["templates"])
            text = template.format(
                subject=rng.choice(data["subjects"]),
                action=rng.choice(data["actions"]),
                target=rng.choice(data["targets"]),
                context=rng.choice(data["contexts"])
            )
            
            # Ensure zero duplication across the dataset
            if text not in seen_texts:
                seen_texts.add(text)
                dataset.append({
                    "id": str(uuid.uuid4()),
                    "text": text,
                    "category": category_name
                })
                generated_count += 1

    # Shuffle the combined dataset deterministically so categories are blended
    rng.shuffle(dataset)
    return dataset

def main():
    os.makedirs('data/processed', exist_ok=True)
    vectors_path = 'data/processed/vectors.npy'
    meta_path = 'data/processed/metadata.jsonl'

    # 1. Generate text records
    dataset = generate_synthetic_dataset(total_records=50000, seed=42)
    texts = [record["text"] for record in dataset]
    
    # 2. Embed and Save
    pipeline = EmbeddingPipeline()
    vectors = pipeline.generate_embeddings(texts)
    pipeline.save_data(vectors, dataset, vectors_path, meta_path)

if __name__ == "__main__":
    main()