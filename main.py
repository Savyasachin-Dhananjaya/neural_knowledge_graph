import contextlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, List, Literal, Optional

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# --- 1. CONFIGURATION ---
MEMORY_FILE_PATH = Path("memory.json")
WL_CONFIG = "l3_supercat"  # Using the powerful Llama 3 distilled model
WL_DIM = 1024  # High precision dimensions
WL_TRUNC = 256  # Truncated for speed (good enough for reranking)

# Global Neural Engine
wl = None


# --- 2. LIFESPAN: STANDARD LOADER ---
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global wl
    print("----------------------------------------------------------------")
    print(f"🧠 Starting Knowledge Graph Server ({WL_CONFIG})...")

    try:
        from wordllama import WordLlama

        print(f"🦙 Loading WordLlama model: {WL_CONFIG}...")
        # Standard easy loading (Auto-downloads if needed)
        wl = WordLlama.load(config=WL_CONFIG, dim=WL_DIM, trunc_dim=WL_TRUNC)
        print("✅ Neural Engine Active!")
    except ImportError:
        print("⚠️  'wordllama' not found. Run: pip install wordllama")
    except Exception as e:
        print(f"❌ Error loading WordLlama: {e}")

    yield
    print("🛑 Shutting down server...")


app = FastAPI(
    title="Knowledge Graph Server",
    version="3.1.0",
    description="Semantic Memory with full Temporal Provenance and complete CRUD.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. DATA MODELS (With Full Provenance) ---


class Observation(BaseModel):
    content: str
    timestamp: str = Field(
        ..., description="Creation time of this specific fact (Immutable)"
    )


class Entity(BaseModel):
    name: str = Field(..., description="Unique name")
    entityType: str = Field(..., description="Category Tag")
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="When this folder was opened",
    )
    last_modified: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="When it was last touched",
    )
    observations: List[Observation] = []


class Relation(BaseModel):
    from_: str = Field(..., alias="from")
    to: str
    relationType: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class KnowledgeGraph(BaseModel):
    entities: List[Entity]
    relations: List[Relation]


# --- 4. REQUEST MODELS ---


class CreateEntitiesRequest(BaseModel):
    entities: List[dict]


class CreateRelationsRequest(BaseModel):
    relations: List[Relation]


class ObservationItem(BaseModel):
    entityName: str
    contents: List[str]


class AddObservationsRequest(BaseModel):
    observations: List[ObservationItem]


class DeleteEntitiesRequest(BaseModel):
    entityNames: List[str]


class DeleteObservationsRequest(BaseModel):
    deletions: List[dict]


class DeleteRelationsRequest(BaseModel):
    relations: List[Relation]


class SearchNodesRequest(BaseModel):
    query: str


class OpenNodesRequest(BaseModel):
    names: List[str]


class RetrieveContextRequest(BaseModel):
    canonical_entity: str
    query_thought: str
    top_k: int = 3


# --- 5. I/O HANDLERS ---


def read_graph_file() -> KnowledgeGraph:
    if not MEMORY_FILE_PATH.exists():
        return KnowledgeGraph(entities=[], relations=[])

    with open(MEMORY_FILE_PATH, "r", encoding="utf-8") as f:
        try:
            content = f.read().strip()
            if not content:
                return KnowledgeGraph(entities=[], relations=[])
            data = json.loads(content)

            entities = []
            for e in data.get("entities", []):
                creation_time = (
                    e.get("created_at")
                    or e.get("last_modified")
                    or datetime.now().isoformat()
                )
                obs_list = []
                for o in e.get("observations", []):
                    if isinstance(o, str):
                        obs_list.append(Observation(content=o, timestamp=creation_time))
                    else:
                        obs_list.append(Observation(**o))

                entities.append(
                    Entity(
                        name=e["name"],
                        entityType=e["entityType"],
                        created_at=creation_time,
                        last_modified=e.get(
                            "last_modified", datetime.now().isoformat()
                        ),
                        observations=obs_list,
                    )
                )

            relations = []
            for r in data.get("relations", []):
                if "created_at" not in r:
                    r["created_at"] = datetime.now().isoformat()
                relations.append(Relation(**r))

            return KnowledgeGraph(entities=entities, relations=relations)
        except json.JSONDecodeError:
            return KnowledgeGraph(entities=[], relations=[])


def save_graph(graph: KnowledgeGraph):
    with open(MEMORY_FILE_PATH, "w", encoding="utf-8") as f:
        try:
            f.write(graph.model_dump_json(indent=2, by_alias=True))
        except AttributeError:
            f.write(graph.json(indent=2, by_alias=True))


# --- 6. ENDPOINTS ---


@app.get("/read_graph", response_model=KnowledgeGraph)
def read_graph():
    return read_graph_file()


@app.post("/create_entities")
def create_entities(req: CreateEntitiesRequest):
    graph = read_graph_file()
    existing_names = {e.name for e in graph.entities}
    current_time = datetime.now().isoformat()

    for item in req.entities:
        name = item.get("name")
        if name and name not in existing_names:
            ent = Entity(
                name=name,
                entityType=item.get("entityType", "General"),
                created_at=current_time,
                last_modified=current_time,
                observations=[],
            )
            if "observations" in item:
                for obs in item["observations"]:
                    if isinstance(obs, str):
                        ent.observations.append(
                            Observation(content=obs, timestamp=current_time)
                        )
            graph.entities.append(ent)
            existing_names.add(name)

    save_graph(graph)
    return {"message": "Entities created"}


@app.post("/create_relations")
def create_relations(req: CreateRelationsRequest):
    graph = read_graph_file()
    existing = {(r.from_, r.to, r.relationType) for r in graph.relations}
    current_time = datetime.now().isoformat()

    for r in req.relations:
        if (r.from_, r.to, r.relationType) not in existing:
            if not r.created_at:
                r.created_at = current_time
            graph.relations.append(r)
            existing.add((r.from_, r.to, r.relationType))
    save_graph(graph)
    return {"message": "Relations created"}


@app.post("/add_observations")
def add_observations(req: AddObservationsRequest):
    graph = read_graph_file()
    current_time = datetime.now().isoformat()
    results = []

    for item in req.observations:
        entity = next((e for e in graph.entities if e.name == item.entityName), None)
        if entity:
            count = 0
            for content_str in item.contents:
                if not any(o.content == content_str for o in entity.observations):
                    entity.observations.append(
                        Observation(content=content_str, timestamp=current_time)
                    )
                    count += 1
            if count > 0:
                entity.last_modified = current_time
                results.append(f"Added {count} to {item.entityName}")

    save_graph(graph)
    return {"results": results}


@app.post("/delete_entities")
def delete_entities(req: DeleteEntitiesRequest):
    graph = read_graph_file()
    graph.entities = [e for e in graph.entities if e.name not in req.entityNames]
    graph.relations = [
        r
        for r in graph.relations
        if r.from_ not in req.entityNames and r.to not in req.entityNames
    ]
    save_graph(graph)
    return {"message": "Deleted entities"}


@app.post("/delete_observations")
def delete_observations(req: DeleteObservationsRequest):
    graph = read_graph_file()
    for item in req.deletions:
        entity = next(
            (e for e in graph.entities if e.name == item.get("entityName")), None
        )
        if entity:
            to_delete = item.get("observations", [])
            entity.observations = [
                o for o in entity.observations if o.content not in to_delete
            ]
            entity.last_modified = datetime.now().isoformat()
    save_graph(graph)
    return {"message": "Deleted observations"}


@app.post("/delete_relations")
def delete_relations(req: DeleteRelationsRequest):
    graph = read_graph_file()
    # Create a set of relations to delete (matching from/to/type)
    to_delete = {(r.from_, r.to, r.relationType) for r in req.relations}

    # Filter list: Keep only relations NOT in the deletion set
    graph.relations = [
        r for r in graph.relations if (r.from_, r.to, r.relationType) not in to_delete
    ]

    save_graph(graph)
    return {"message": "Deleted relations"}


@app.post("/open_nodes")
def open_nodes(req: OpenNodesRequest):
    graph = read_graph_file()
    target_names = set(req.names)

    # 1. Get the Entities
    entities = [e for e in graph.entities if e.name in target_names]
    found_names = {e.name for e in entities}

    # 2. Get connected relations (Incoming OR Outgoing)
    relations = [
        r for r in graph.relations if r.from_ in found_names or r.to in found_names
    ]

    return KnowledgeGraph(entities=entities, relations=relations)


@app.post("/search_nodes")
def search_nodes(req: SearchNodesRequest):
    graph = read_graph_file()
    query = req.query.lower()
    matched_entities = []
    found_names = set()

    for e in graph.entities:
        if (
            query in e.name.lower()
            or query in e.entityType.lower()
            or any(query in o.content.lower() for o in e.observations)
        ):
            matched_entities.append(e)
            found_names.add(e.name)

    matched_relations = [
        r for r in graph.relations if r.from_ in found_names and r.to in found_names
    ]
    return KnowledgeGraph(entities=matched_entities, relations=matched_relations)


@app.post("/retrieve_context")
def retrieve_context(req: RetrieveContextRequest):
    if not wl:
        return {"error": "WordLlama engine not loaded."}

    graph = read_graph_file()
    target_tag = req.canonical_entity.lower()

    candidate_docs = []
    doc_map = {}

    for entity in graph.entities:
        if target_tag in entity.entityType.lower():
            for obs in entity.observations:
                candidate_docs.append(obs.content)
                doc_map[obs.content] = obs

    if not candidate_docs:
        return {
            "tag": req.canonical_entity,
            "results": [],
            "message": "No memories found.",
        }

    try:
        ranked_results = wl.rank(req.query_thought, candidate_docs, sort=True)

        top_results = []
        for text, score in ranked_results[: req.top_k]:
            original = doc_map.get(text)
            top_results.append(
                {
                    "content": text,
                    "score": round(float(score), 4),
                    "created_at": original.timestamp if original else "",
                }
            )

        return {
            "tag": req.canonical_entity,
            "query": req.query_thought,
            "results": top_results,
        }
    except Exception as e:
        return {"error": f"Reranking failed: {str(e)}"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9041)
