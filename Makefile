# Define URLs for supported knowledge graphs
WD_URL=https://qlever.cs.uni-freiburg.de/api/wikidata
FB_URL=https://qlever.cs.uni-freiburg.de/api/freebase
DBPEDIA_URL=https://qlever.cs.uni-freiburg.de/api/dbpedia
DBLP_URL=https://qlever.cs.uni-freiburg.de/api/dblp
UNIPROT_URL=https://qlever.cs.uni-freiburg.de/api/uniprot
OSM_URL=https://qlever.cs.uni-freiburg.de/api/osm-planet
ORKG_URL=https://qlever.cs.uni-freiburg.de/api/orkg
IMDB_URL=https://qlever.cs.uni-freiburg.de/api/imdb

# Some QLever API parameters
QLEVER_TIMEOUT=12h
QLEVER_ACCESS_TOKEN=null

# Defaults for the types of indices to build
ENT_SEARCH_INDEX=prefix
ENT_ARGS=
PROP_SEARCH_INDEX=similarity
PROP_ARGS=

# Generic argument placeholder for various targets
ARGS=

all:
	@echo "This target does nothing, you most likely want to use \
	the pre-built indices to run GRASP; follow the README to do so. \n\
	If you want to build an index for a supported knowledge graph yourself, e.g. Wikidata, \
	start with 'make wikidata-kg-data QLEVER_ACCESS_TOKEN=...', \
	followed by 'make wikidata-kg-indices'; analogously for other supported knowledge graphs. \n\
	There is also a generic target 'make generic-kg-data KG_NAME=... KG_URL=... QLEVER_ACCESS_TOKEN=...' \
	and 'make generic-kg-indices KG_NAME=...' to build indices for any knowledge graph."

benchmarks: wikidata-benchmarks \
	freebase-benchmarks \
	dbpedia-benchmarks \
	dblp-benchmarks \
	orkg-benchmarks

wikidata-benchmarks:
	@python scripts/prepare_benchmark.py \
	--wikidata-simple-questions \
	--out-dir data/benchmark/wikidata/simplequestions \
	$(ARGS)
	@python scripts/prepare_benchmark.py \
	--lc-quad2-wikidata \
	--out-dir data/benchmark/wikidata/lcquad2 \
	$(ARGS)
	@python scripts/prepare_benchmark.py \
	--qald-10 \
	--out-dir data/benchmark/wikidata/qald10 \
	$(ARGS)
	@python scripts/prepare_benchmark.py \
	--qald-7 data/raw/qald-7 \
	--out-dir data/benchmark/wikidata/qald7 \
	$(ARGS)
	@python scripts/prepare_benchmark.py \
	--wwq data/raw/wikiwebquestions \
	--out-dir data/benchmark/wikidata/wwq \
	$(ARGS)
	@python scripts/prepare_benchmark.py \
	--spinach data/raw/spinach \
	--out-dir data/benchmark/wikidata/spinach \
	$(ARGS)

freebase-benchmarks:
	@python scripts/prepare_benchmark.py \
	--wqsp \
	--out-dir data/benchmark/freebase/wqsp \
	$(ARGS)
	@python scripts/prepare_benchmark.py \
	--cwq \
	--out-dir data/benchmark/freebase/cwq \
	$(ARGS)
	@python scripts/prepare_benchmark.py \
	--freebase-simple-questions data/raw/simplequestions-freebase \
	--out-dir data/benchmark/freebase/simplequestions \
	$(ARGS)

dbpedia-benchmarks:
	@python scripts/prepare_benchmark.py \
	--lc-quad1-dbpedia \
	--out-dir data/benchmark/dbpedia/lcquad \
	$(ARGS)
	@python scripts/prepare_benchmark.py \
	--qald-7-dbpedia data/raw/qald7 \
	--out-dir data/benchmark/dbpedia/qald7 \
	$(ARGS)
	@python scripts/prepare_benchmark.py \
	--qald-9 \
	--out-dir data/benchmark/dbpedia/qald9 \
	$(ARGS)
	@python scripts/prepare_benchmark.py \
	--dbpedia-simple-questions data/raw/simplequestions-dbpedia \
	--out-dir data/benchmark/dbpedia/simplequestions \
	$(ARGS)

dblp-benchmarks:
	@python scripts/prepare_benchmark.py \
	--dblp-quad \
	--out-dir data/benchmark/dblp/dblp-quad \
	$(ARGS)

orkg-benchmarks:
	@python scripts/prepare_benchmark.py \
	--sci-qa \
	--out-dir data/benchmark/orkg-2023/sci-qa \
	$(ARGS)

example-indices:
	@for f in $(wildcard data/benchmark/*/*/train.jsonl); do \
		python scripts/build_example_index.py \
		$$f \
		`dirname $$f`/train.example-index \
		$(ARGS); \
	done

kg-indices: \
	wikidata-kg-indices \
	freebase-kg-indices \
	dbpedia-kg-indices \
	uniprot-kg-indices \
	osm-planet-kg-indices \
	dblp-kg-indices \
	orkg-kg-indices \
	imdb-kg-indices

KGS=wikidata freebase dbpedia uniprot osm-planet dblp orkg imdb

kg-prefixes:
	@for kg in $(KGS); do \
		mkdir -p data/kg-index/$$kg; \
		curl -s https://qlever.cs.uni-freiburg.de/api/prefixes/$$kg \
		| python scripts/qlever_prefixes_to_json.py \
		> data/kg-index/$$kg/prefixes.json; \
	done

KG_NAME=generic
KG_URL=https://qlever.cs.uni-freiburg.de/api/generic

generic-kg-data:
	# entities
	# representative query over IMDb:
	# https://qlever.cs.uni-freiburg.de/imdb/2Aop9v
	@mkdir -p data/kg-index/$(KG_NAME)/entities
	@curl -s $(KG_URL) -H "Accept: text/csv" \
	--data-urlencode query="PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX skos: <http://www.w3.org/2004/02/skos/core#> PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> SELECT (SAMPLE(?label) AS ?main_label) (SUM(?count) AS ?score) (GROUP_CONCAT(DISTINCT ?alias; SEPARATOR=\";;;\") AS ?synonyms) ?id (GROUP_CONCAT(DISTINCT ?info; SEPARATOR=\";;;\") AS ?infos) WHERE { { SELECT ?id (COUNT(?id) AS ?count) WHERE { ?id ?p ?o } GROUP BY ?id } UNION { SELECT ?id (COUNT(?id) AS ?count) WHERE { ?s ?p ?id . FILTER(ISIRI(?id)) } GROUP BY ?id } OPTIONAL { ?id rdfs:label ?label_en FILTER(LANG(?label_en) = \"en\") } OPTIONAL { ?id rdfs:label ?label_plain FILTER(LANG(?label_plain) = \"\" || DATATYPE(?label_plain) = xsd:string) } BIND(COALESCE(?label_en, ?label_plain) AS ?label) OPTIONAL { ?id skos:altLabel ?alias_en FILTER(LANG(?alias_en) = \"en\") } OPTIONAL { ?id skos:altLabel ?alias_plain FILTER(LANG(?alias_plain) = \"\" || DATATYPE(?alias_plain) = xsd:string) } BIND(COALESCE(?alias_en, ?alias_plain) AS ?alias) OPTIONAL { { ?id rdfs:comment ?comment_en FILTER(LANG(?comment_en) = \"en\") } UNION { ?id rdfs:comment ?comment_plain FILTER(LANG(?comment_plain) = \"\" || DATATYPE(?comment_plain) = xsd:string) } BIND(COALESCE(?comment_en, ?comment_plain) AS ?info) } OPTIONAL { ?id rdfs:domain ?d . OPTIONAL { ?d rdfs:label ?domain_en FILTER(LANG(?domain_en) = \"en\") } OPTIONAL { ?d rdfs:label ?domain_plain FILTER(LANG(?domain_plain) = \"\" || DATATYPE(?domain_plain) = xsd:string) } BIND(COALESCE(?domain_en, ?domain_plain) AS ?domain) BIND(CONCAT(\"has domain \", ?domain) AS ?info) } OPTIONAL { ?id rdfs:range ?r . OPTIONAL { ?r rdfs:label ?range_en FILTER(LANG(?range_en) = \"en\") } OPTIONAL { ?r rdfs:label ?range_plain FILTER(LANG(?range_plain) = \"\" || DATATYPE(?range_plain) = xsd:string) } BIND(COALESCE(?range_en, ?range_plain) AS ?range) BIND(CONCAT(\"has range \", ?range) AS ?info) } OPTIONAL { ?id rdf:type ?t . OPTIONAL { ?t rdfs:label ?type_en FILTER(LANG(?type_en) = \"en\") } OPTIONAL { ?t rdfs:label ?type_plain FILTER(LANG(?type_plain) = \"\" || DATATYPE(?type_plain) = xsd:string) } BIND(COALESCE(?type_en, ?type_plain) AS ?type) BIND(CONCAT(\"is a \", ?type) AS ?info) } OPTIONAL { ?id rdfs:subClassOf ?sc . OPTIONAL { ?sc rdfs:label ?superclass_en FILTER(LANG(?superclass_en) = \"en\") } OPTIONAL { ?sc rdfs:label ?superclass_plain FILTER(LANG(?superclass_plain) = \"\" || DATATYPE(?superclass_plain) = xsd:string) } BIND(COALESCE(?superclass_en, ?superclass_plain) AS ?superclass) BIND(CONCAT(\"is subclass of \", ?superclass) AS ?info) } } GROUP BY ?id ORDER BY DESC(?score)" \
	--data-urlencode timeout=$(QLEVER_TIMEOUT) \
	--data-urlencode access-token=$(QLEVER_ACCESS_TOKEN) \
	| python scripts/prepare_kg_data.py \
	--kg $(KG_NAME) \
	> data/kg-index/$(KG_NAME)/entities/data.tsv

	@python scripts/build_kg_data_and_mapping.py \
	data/kg-index/$(KG_NAME)/entities/data.tsv \
	data/kg-index/$(KG_NAME)/entities/offsets.bin \
	data/kg-index/$(KG_NAME)/entities/mapping.bin \
	--overwrite

	# properties
	# representative query over IMDb:
	# https://qlever.cs.uni-freiburg.de/imdb/72RX9m
	@mkdir -p data/kg-index/$(KG_NAME)/properties
	@curl -s $(KG_URL) -H "Accept: text/csv" \
	--data-urlencode query="PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> SELECT (SAMPLE(?label) AS ?main_label) ?score (GROUP_CONCAT(DISTINCT ?alias; SEPARATOR=\";;;\") AS ?synonyms) ?id (GROUP_CONCAT(DISTINCT ?info; SEPARATOR=\";;;\") AS ?infos) WHERE { { SELECT ?id (COUNT(?id) AS ?score) WHERE { ?s ?id ?o } GROUP BY ?id } OPTIONAL { ?id rdfs:label ?label_en FILTER(LANG(?label_en) = \"en\") } OPTIONAL { ?id rdfs:label ?label_plain FILTER(LANG(?label_plain) = \"\" || DATATYPE(?label_plain) = xsd:string) } BIND(COALESCE(?label_en, ?label_plain) AS ?label) BIND(\"\" AS ?alias) OPTIONAL { { ?id rdfs:comment ?comment_en FILTER(LANG(?comment_en) = \"en\") } UNION { ?id rdfs:comment ?comment_plain FILTER(LANG(?comment_plain) = \"\" || DATATYPE(?comment_plain) = xsd:string) } BIND(COALESCE(?comment_en, ?comment_plain) AS ?info) } OPTIONAL { ?id rdfs:domain ?d . OPTIONAL { ?d rdfs:label ?domain_en FILTER(LANG(?domain_en) = \"en\") } OPTIONAL { ?d rdfs:label ?domain_plain FILTER(LANG(?domain_plain) = \"\" || DATATYPE(?domain_plain) = xsd:string) } BIND(COALESCE(?domain_en, ?domain_plain) AS ?domain) BIND(CONCAT(\"has domain \", ?domain) AS ?info) } OPTIONAL { ?id rdfs:range ?r . OPTIONAL { ?r rdfs:label ?range_en FILTER(LANG(?range_en) = \"en\") } OPTIONAL { ?r rdfs:label ?range_plain FILTER(LANG(?range_plain) = \"\" || DATATYPE(?range_plain) = xsd:string) } BIND(COALESCE(?range_en, ?range_plain) AS ?range) BIND(CONCAT(\"has range \", ?range) AS ?info) } OPTIONAL { ?id rdf:type ?t . OPTIONAL { ?t rdfs:label ?type_en FILTER(LANG(?type_en) = \"en\") } OPTIONAL { ?t rdfs:label ?type_plain FILTER(LANG(?type_plain) = \"\" || DATATYPE(?type_plain) = xsd:string) } BIND(COALESCE(?type_en, ?type_plain) AS ?type) BIND(CONCAT(\"is a \", ?type) AS ?info) } OPTIONAL { ?id rdfs:subPropertyOf ?sp . OPTIONAL { ?sp rdfs:label ?superclass_en FILTER(LANG(?superclass_en) = \"en\") } OPTIONAL { ?sp rdfs:label ?superclass_plain FILTER(LANG(?superclass_plain) = \"\" || DATATYPE(?superclass_plain) = xsd:string) } BIND(COALESCE(?superclass_en, ?superclass_plain) AS ?superclass) BIND(CONCAT(\"is subproperty of \", ?superclass) AS ?info) } } GROUP BY ?score ?id ORDER BY DESC(?score)" \
	--data-urlencode timeout=$(QLEVER_TIMEOUT) \
	--data-urlencode access-token=$(QLEVER_ACCESS_TOKEN) \
	| python scripts/prepare_kg_data.py \
	--kg $(KG_NAME) \
	--add-id-as-synonym \
	> data/kg-index/$(KG_NAME)/properties/data.tsv

	@python scripts/build_kg_data_and_mapping.py \
	data/kg-index/$(KG_NAME)/properties/data.tsv \
	data/kg-index/$(KG_NAME)/properties/offsets.bin \
	data/kg-index/$(KG_NAME)/properties/mapping.bin \
	--overwrite

generic-kg-indices:
	@python scripts/build_kg_index.py \
	data/kg-index/$(KG_NAME)/entities \
	data/kg-index/$(KG_NAME)/entities/$(ENT_SEARCH_INDEX) \
	--type $(ENT_SEARCH_INDEX) $(ENT_ARGS)

	@python scripts/build_kg_index.py \
	data/kg-index/$(KG_NAME)/properties \
	data/kg-index/$(KG_NAME)/properties/$(PROP_SEARCH_INDEX) \
	--type $(PROP_SEARCH_INDEX) $(PROP_ARGS)
	
wikidata-kg-data:
	# wikidata entities
	# https://qlever.cs.uni-freiburg.de/wikidata/7ECsCI
	@mkdir -p data/kg-index/wikidata/entities
	@curl -s $(WD_URL) -H "Accept: text/csv" \
	--data-urlencode query="PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX skos: <http://www.w3.org/2004/02/skos/core#> PREFIX schema: <http://schema.org/> PREFIX wikibase: <http://wikiba.se/ontology#> PREFIX wd: <http://www.wikidata.org/entity/> PREFIX wdt: <http://www.wikidata.org/prop/direct/> SELECT ?label ?score (GROUP_CONCAT(DISTINCT ?alias; SEPARATOR=\";;;\") AS ?synonyms) ?id (GROUP_CONCAT(DISTINCT ?info; SEPARATOR=\";;;\") AS ?infos) WHERE { { ?id @en@rdfs:label ?label } MINUS { ?id wdt:P31/wdt:P279* wd:Q17442446 } OPTIONAL { ?id @en@skos:altLabel ?alias } OPTIONAL { ?id ^schema:about/wikibase:sitelinks ?score } OPTIONAL { { ?id @en@schema:description ?info } UNION { ?id wdt:P279/@en@rdfs:label ?subclass_label . BIND(CONCAT(\"subclass of \", ?sublcass_label) AS ?info) } UNION { ?id wdt:P31/@en@rdfs:label ?inst_label . BIND(CONCAT(\"instance of \", ?inst_label) AS ?info) } UNION { ?id wdt:P106/@en@rdfs:label ?info } } } GROUP BY ?label ?score ?id ORDER BY DESC(?score)" \
	--data-urlencode timeout=$(QLEVER_TIMEOUT) \
	--data-urlencode access-token=$(QLEVER_ACCESS_TOKEN) \
	| python scripts/prepare_kg_data.py \
	--kg wikidata \
	> data/kg-index/wikidata/entities/data.tsv

	@python scripts/build_kg_data_and_mapping.py \
	data/kg-index/wikidata/entities/data.tsv \
	data/kg-index/wikidata/entities/offsets.bin \
	data/kg-index/wikidata/entities/mapping.bin \
	--overwrite

	# wikidata properties
	# https://qlever.cs.uni-freiburg.de/wikidata/Opl9T4
	@curl -s $(WD_URL) -H "Accept: text/csv" \
	--data-urlencode query="PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX skos: <http://www.w3.org/2004/02/skos/core#> PREFIX schema: <http://schema.org/> PREFIX wikibase: <http://wikiba.se/ontology#> PREFIX wd: <http://www.wikidata.org/entity/> PREFIX wdt: <http://www.wikidata.org/prop/direct/> SELECT ?label ?score (GROUP_CONCAT(DISTINCT ?alias; SEPARATOR=\";;;\") AS ?synonyms) ?id (GROUP_CONCAT(DISTINCT ?info; SEPARATOR=\";;;\") AS ?infos) WHERE { { SELECT ?p (COUNT(?p) AS ?count) WHERE { ?s ?p ?o } GROUP BY ?p } { SELECT (MAX(?count) AS ?max) WHERE { SELECT (COUNT(?p) AS ?count) WHERE { ?s ?p ?o . ?p ^wikibase:claim/wikibase:propertyType wikibase:ExternalId } GROUP BY ?p } } ?id wikibase:claim ?p . ?id @en@rdfs:label ?label . OPTIONAL { ?id @en@skos:altLabel ?alias } ?id wikibase:propertyType ?type . BIND(IF(?type = wikibase:ExternalId, ?count, ?max + 1 + ?count) AS ?score) OPTIONAL { { ?id @en@schema:description ?info } UNION { ?id wdt:P1647/@en@rdfs:label ?sub_label . BIND(CONCAT(\"subproperty of \", ?sub_label) AS ?info) } } } GROUP BY ?label ?score ?id ORDER BY DESC(?score)" \
	--data-urlencode timeout=$(QLEVER_TIMEOUT) \
	--data-urlencode access-token=$(QLEVER_ACCESS_TOKEN) \
	| python scripts/prepare_kg_data.py \
	--kg wikidata \
	--add-id-as-synonym \
	> data/kg-index/wikidata/properties/data.tsv

	@python scripts/build_kg_data_and_mapping.py \
	data/kg-index/wikidata/properties/data.tsv \
	data/kg-index/wikidata/properties/offsets.bin \
	data/kg-index/wikidata/properties/mapping.bin \
	--overwrite

wikidata-kg-indices:
	@python scripts/build_kg_index.py \
	data/kg-index/wikidata/entities \
	data/kg-index/wikidata/entities/$(ENT_SEARCH_INDEX) \
	--type $(ENT_SEARCH_INDEX) $(ENT_ARGS)

	@python scripts/build_kg_index.py \
	data/kg-index/wikidata/properties \
	data/kg-index/wikidata/properties/$(PROP_SEARCH_INDEX) \
	--type $(PROP_SEARCH_INDEX) $(PROP_ARGS)

imdb-kg-data:
	# imdb entities
	# https://qlever.cs.uni-freiburg.de/imdb/iX7WAZ
	@mkdir -p data/kg-index/imdb/entities
	@curl -s $(IMDB_URL) -H "Accept: text/csv" \
	--data-urlencode query="PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX skos: <http://www.w3.org/2004/02/skos/core#> SELECT ?label (SUM(?count) AS ?score) (GROUP_CONCAT(DISTINCT ?alias; SEPARATOR=\";;;\") AS ?synonyms) ?id (GROUP_CONCAT(DISTINCT ?info; SEPARATOR=\";;;\") AS ?infos) WHERE { { SELECT ?id (COUNT(?id) AS ?count) WHERE { ?id ?p ?o } GROUP BY ?id } UNION { SELECT ?id (COUNT(?id) AS ?count) WHERE { ?s ?p ?id . FILTER(ISIRI(?id)) } GROUP BY ?id } OPTIONAL { ?id rdfs:label ?label } OPTIONAL { ?id @en@skos:altLabel ?alias } BIND(\"\" AS ?info) } GROUP BY ?label ?id ORDER BY DESC(?score)" \
	--data-urlencode timeout=$(QLEVER_TIMEOUT) \
	--data-urlencode access-token=$(QLEVER_ACCESS_TOKEN) \
	| python scripts/prepare_kg_data.py \
	--kg imdb \
	> data/kg-index/imdb/entities/data.tsv

	@python scripts/build_kg_data_and_mapping.py \
	data/kg-index/imdb/entities/data.tsv \
	data/kg-index/imdb/entities/offsets.bin \
	data/kg-index/imdb/entities/mapping.bin \
	--overwrite

	# imdb properties
	# https://qlever.cs.uni-freiburg.de/imdb/apnR5N
	@mkdir -p data/kg-index/imdb/properties
	@curl -s $(IMDB_URL) -H "Accept: text/csv" \
	--data-urlencode query="PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX skos: <http://www.w3.org/2004/02/skos/core#> SELECT ?label ?score (GROUP_CONCAT(DISTINCT ?alias; SEPARATOR=\";;;\") AS ?synonyms) ?id (GROUP_CONCAT(DISTINCT ?info; SEPARATOR=\";;;\") AS ?infos) WHERE { { SELECT ?id (COUNT(?id) AS ?score) WHERE { ?s ?id ?o } GROUP BY ?id } BIND(\"\" AS ?label) BIND(\"\" AS ?alias) BIND(\"\" AS ?info) } GROUP BY ?label ?score ?id ORDER BY DESC(?score)" \
	--data-urlencode timeout=$(QLEVER_TIMEOUT) \
	--data-urlencode access-token=$(QLEVER_ACCESS_TOKEN) \
	| python scripts/prepare_kg_data.py \
	--kg imdb \
	--add-id-as-synonym \
	> data/kg-index/imdb/properties/data.tsv

	@python scripts/build_kg_data_and_mapping.py \
	data/kg-index/imdb/properties/data.tsv \
	data/kg-index/imdb/properties/offsets.bin \
	data/kg-index/imdb/properties/mapping.bin \
	--overwrite

imdb-kg-indices:
	@python scripts/build_kg_index.py \
	data/kg-index/imdb/entities \
	data/kg-index/imdb/entities/$(ENT_SEARCH_INDEX) \
	--type $(ENT_SEARCH_INDEX) $(ENT_ARGS)

	@python scripts/build_kg_index.py \
	data/kg-index/imdb/properties \
	data/kg-index/imdb/properties/$(PROP_SEARCH_INDEX) \
	--type $(PROP_SEARCH_INDEX) $(PROP_ARGS)

uniprot-kg-data:
	# uniprot entities
	# https://qlever.cs.uni-freiburg.de/uniprot/TPzsht
	@mkdir -p data/kg-index/uniprot/entities
	@curl -s $(UNIPROT_URL) -H "Accept: text/csv" \
	--data-urlencode query="PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX skos: <http://www.w3.org/2004/02/skos/core#> PREFIX schema: <http://schema.org/> PREFIX wikibase: <http://wikiba.se/ontology#> PREFIX wd: <http://www.wikidata.org/entity/> PREFIX wdt: <http://www.wikidata.org/prop/direct/> PREFIX up: <http://purl.uniprot.org/core/> PREFIX owl: <http://www.w3.org/2002/07/owl#> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> SELECT ?label ?score (GROUP_CONCAT(DISTINCT ?alias; SEPARATOR=\";;;\") AS ?synonyms) ?id (GROUP_CONCAT(DISTINCT ?info; SEPARATOR=\";;;\") AS ?infos) WHERE { { ?id rdf:type owl:Class } UNION { ?id rdf:type owl:DatatypeProperty } UNION { ?id rdf:type up:Taxon } UNION { ?id rdf:type up:Database } UNION { ?id rdf:type up:Concept } UNION { ?id rdf:type up:Enzyme } UNION { ?id rdf:type up:Proteome } UNION { ?id rdf:type up:Chromosome } OPTIONAL { ?id rdfs:label ?label } OPTIONAL { { ?id up:scientificName ?alias } UNION { ?id up:otherName ?alias } UNION { ?id up:commonName ?alias } UNION { ?id up:mnemonic ?alias } UNION { ?id up:synonym ?alias } UNION { ?id <http://www.geneontology.org/formats/oboInOwl#hasExactSynonym> ?alias } UNION { ?id <http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym> ?alias } } BIND(0 AS ?score) OPTIONAL { { ?id rdfs:comment ?info } UNION { ?id <http://purl.obolibrary.org/obo/IAO_0000115> ?info } } } GROUP BY ?label ?score ?id ORDER BY DESC(?score)" \
	--data-urlencode timeout=$(QLEVER_TIMEOUT) \
	--data-urlencode access-token=$(QLEVER_ACCESS_TOKEN) \
	| python scripts/prepare_kg_data.py \
	--kg uniprot \
	> data/kg-index/uniprot/entities/data.tsv

	@python scripts/build_kg_data_and_mapping.py \
	data/kg-index/uniprot/entities/data.tsv \
	data/kg-index/uniprot/entities/offsets.bin \
	data/kg-index/uniprot/entities/mapping.bin \
	--overwrite

	# uniprot properties
	# https://qlever.cs.uni-freiburg.de/uniprot/ArNNbu
	@mkdir -p data/kg-index/uniprot/properties
	@curl -s $(UNIPROT_URL) -H "Accept: text/csv" \
	--data-urlencode query="PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX skos: <http://www.w3.org/2004/02/skos/core#> SELECT ?label ?score (GROUP_CONCAT(DISTINCT ?alias; SEPARATOR=\";;;\") AS ?synonyms) ?id (GROUP_CONCAT(DISTINCT ?info; SEPARATOR=\";;;\") AS ?infos) WHERE { { SELECT ?id (COUNT(?id) AS ?score) WHERE { ?s ?id ?o } GROUP BY ?id } OPTIONAL { ?id rdfs:label ?label } BIND(\"\" AS ?alias) . OPTIONAL { { ?id rdfs:comment ?info } UNION { ?id rdfs:domain/rdfs:label ?domain . BIND(CONCAT(\"has domain \", ?domain) AS ?info) } UNION { ?id rdfs:range/rdfs:label ?range . BIND(CONCAT(\"has range \", ?range) AS ?info) } } } GROUP BY ?label ?score ?id ORDER BY DESC(?score)" \
	--data-urlencode timeout=$(QLEVER_TIMEOUT) \
	--data-urlencode access-token=$(QLEVER_ACCESS_TOKEN) \
	| python scripts/prepare_kg_data.py \
	--kg uniprot \
	--add-id-as-synonym \
	> data/kg-index/uniprot/properties/data.tsv

	@python scripts/build_kg_data_and_mapping.py \
	data/kg-index/uniprot/properties/data.tsv \
	data/kg-index/uniprot/properties/offsets.bin \
	data/kg-index/uniprot/properties/mapping.bin \
	--overwrite

uniprot-kg-indices:
	@python scripts/build_kg_index.py \
	data/kg-index/uniprot/entities \
	data/kg-index/uniprot/entities/$(ENT_SEARCH_INDEX) \
	--type $(ENT_SEARCH_INDEX) $(ENT_ARGS)

	@python scripts/build_kg_index.py \
	data/kg-index/uniprot/properties \
	data/kg-index/uniprot/properties/$(PROP_SEARCH_INDEX) \
	--type $(PROP_SEARCH_INDEX) $(PROP_ARGS)

osm-planet-kg-data:
	# osm entities
	# https://qlever.cs.uni-freiburg.de/osm-planet/aRuan8
	@mkdir -p data/kg-index/osm-planet/entities
	@curl -s $(OSM_URL) -H "Accept: text/csv" \
	--data-urlencode query="PREFIX osmkey: <https://www.openstreetmap.org/wiki/Key:> PREFIX osm2rdfkey: <https://osm2rdf.cs.uni-freiburg.de/rdf/key#> SELECT ?label ?score (GROUP_CONCAT(DISTINCT ?alias; SEPARATOR=\";;;\") AS ?synonyms) ?id (GROUP_CONCAT(DISTINCT ?info; SEPARATOR=\";;;\") AS ?infos) WHERE { ?id osmkey:name ?label . OPTIONAL { { ?id osmkey:short_name ?alias } UNION { ?id osmkey:name:en ?alias } UNION { ?id osmkey:name:de ?alias } } OPTIONAL { ?id osm2rdfkey:wikidata ?score } OPTIONAL { { ?id osmkey:admin_title ?info } UNION { ?id osmkey:description ?info } UNION { ?id osmkey:amenity ?info } } } GROUP BY ?label ?score ?id ORDER BY DESC(?score)" \
	--data-urlencode timeout=$(QLEVER_TIMEOUT) \
	--data-urlencode access-token=$(QLEVER_ACCESS_TOKEN) \
	| python scripts/prepare_kg_data.py \
	--osm-planet-entities \
	--kg osm-planet \
	> data/kg-index/osm-planet/entities/data.tsv
	# resort based on score (keep header, sort rest)
	(head -n1 data/kg-index/osm-planet/entities/data.tsv && \
	tail -n+2 data/kg-index/osm-planet/entities/data.tsv | sort -t'	' -k2,2nr) \
	> data/kg-index/osm-planet/entities/data.sorted.tsv
	mv data/kg-index/osm-planet/entities/data.sorted.tsv data/kg-index/osm-planet/entities/data.tsv

	@python scripts/build_kg_data_and_mapping.py \
	data/kg-index/osm-planet/entities/data.tsv \
	data/kg-index/osm-planet/entities/offsets.bin \
	data/kg-index/osm-planet/entities/mapping.bin \
	--overwrite

	# osm properties
	# https://qlever.cs.uni-freiburg.de/osm-planet/8TQ77U
	@mkdir -p data/kg-index/osm-planet/properties
	@curl -s $(OSM_URL) -H "Accept: text/csv" \
	--data-urlencode query="PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX skos: <http://www.w3.org/2004/02/skos/core#> PREFIX schema: <http://schema.org/> PREFIX wikibase: <http://wikiba.se/ontology#> PREFIX wd: <http://www.wikidata.org/entity/> PREFIX wdt: <http://www.wikidata.org/prop/direct/> SELECT ?label ?score (GROUP_CONCAT(DISTINCT ?alias; SEPARATOR=\";;;\") AS ?synonyms) ?id (GROUP_CONCAT(DISTINCT ?info; SEPARATOR=\";;;\") AS ?infos) WHERE { { SELECT ?id (COUNT(?id) AS ?score) WHERE { ?s ?id ?o } GROUP BY ?id } BIND(\"\" AS ?label) BIND(\"\" AS ?alias) BIND(\"\" AS ?info) } GROUP BY ?label ?score ?id ORDER BY DESC(?score)" \
	--data-urlencode timeout=$(QLEVER_TIMEOUT) \
	--data-urlencode access-token=$(QLEVER_ACCESS_TOKEN) \
	| python scripts/prepare_kg_data.py \
	--kg osm-planet \
	--add-id-as-synonym \
	> data/kg-index/osm-planet/properties/data.tsv

	@python scripts/build_kg_data_and_mapping.py \
	data/kg-index/osm-planet/properties/data.tsv \
	data/kg-index/osm-planet/properties/offsets.bin \
	data/kg-index/osm-planet/properties/mapping.bin \
	--overwrite

osm-planet-kg-indices:
	@python scripts/build_kg_index.py \
	data/kg-index/osm-planet/entities \
	data/kg-index/osm-planet/entities/$(ENT_SEARCH_INDEX) \
	--type $(ENT_SEARCH_INDEX) $(ENT_ARGS)

	@python scripts/build_kg_index.py \
	data/kg-index/osm-planet/properties \
	data/kg-index/osm-planet/properties/$(PROP_SEARCH_INDEX) \
	--type $(PROP_SEARCH_INDEX) $(PROP_ARGS)

freebase-kg-data:
	# freebase entities
	# https://qlever.cs.uni-freiburg.de/freebase/MMaV2z
	@mkdir -p data/kg-index/freebase/entities
	@curl -s $(FB_URL) -H "Accept: text/csv" \
	--data-urlencode query="PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX fb: <http://rdf.freebase.com/ns/> PREFIX skos: <http://www.w3.org/2004/02/skos/core#> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> SELECT ?label ?score (GROUP_CONCAT(DISTINCT ?alias; SEPARATOR=\";;;\") AS ?synonyms) ?id (GROUP_CONCAT(DISTINCT ?info; SEPARATOR=\";;;\") AS ?infos) WHERE { { SELECT ?id (COUNT(?p) AS ?score) WHERE { ?id ?p ?o } GROUP BY ?id } ?id @en@fb:type.object.name ?label . OPTIONAL { ?id @en@fb:common.topic.alias ?alias } OPTIONAL { { ?id @en@fb:common.topic.description ?info } UNION { ?id fb:common.topic.notable_types/@en@fb:type.object.name ?type_label . BIND(CONCAT(\"has type \", ?type_label) AS ?info) } UNION { ?id fb:freebase.type_hints.mediator ?mediator . FILTER(?mediator = \"true\") . BIND(\"is compound value type / mediator type\" AS ?info) } } } GROUP BY ?label ?score ?id ORDER BY DESC(?score)" \
	--data-urlencode timeout=$(QLEVER_TIMEOUT) \
	--data-urlencode access-token=$(QLEVER_ACCESS_TOKEN) \
	| python scripts/prepare_kg_data.py \
	--kg freebase \
	> data/kg-index/freebase/entities/data.tsv

	@python scripts/build_kg_data_and_mapping.py \
	data/kg-index/freebase/entities/data.tsv \
	data/kg-index/freebase/entities/offsets.bin \
	data/kg-index/freebase/entities/mapping.bin \
	--overwrite

	# freebase properties
	# https://qlever.cs.uni-freiburg.de/freebase/79cmBi
	@mkdir -p data/kg-index/freebase/properties
	@curl -s $(FB_URL) -H "Accept: text/csv" \
	--data-urlencode query="PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX fb: <http://rdf.freebase.com/ns/> PREFIX skos: <http://www.w3.org/2004/02/skos/core#> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> SELECT ?label ?score (GROUP_CONCAT(DISTINCT ?alias; SEPARATOR=\";;;\") AS ?synonyms) ?id (GROUP_CONCAT(DISTINCT ?info; SEPARATOR=\";;;\") AS ?infos) WHERE { { SELECT ?id (COUNT(?id) AS ?score) WHERE { ?s ?id ?o } GROUP BY ?id } ?id @en@fb:type.object.name ?label . ?id fb:type.object.type fb:type.property . OPTIONAL { ?id @en@fb:common.topic.alias ?alias } OPTIONAL { { ?id fb:type.property.schema/@en@rdfs:label ?schema_label . BIND(CONCAT(\"part of \", ?schema_label, \" schema\") AS ?info) } UNION { ?id fb:type.property.expected_type/@en@rdfs:label ?type_label . BIND(CONCAT(\"links to type \", ?type_label) AS ?info) } } } GROUP BY ?label ?score ?id ORDER BY DESC(?score)" \
	--data-urlencode timeout=$(QLEVER_TIMEOUT) \
	--data-urlencode access-token=$(QLEVER_ACCESS_TOKEN) \
	| python scripts/prepare_kg_data.py \
	--kg freebase \
	--add-id-as-synonym \
	> data/kg-index/freebase/properties/data.tsv

	@python scripts/build_kg_data_and_mapping.py \
	data/kg-index/freebase/properties/data.tsv \
	data/kg-index/freebase/properties/offsets.bin \
	data/kg-index/freebase/properties/mapping.bin \
	--overwrite

freebase-kg-indices:
	@python scripts/build_kg_index.py \
	data/kg-index/freebase/entities \
	data/kg-index/freebase/entities/$(ENT_SEARCH_INDEX) \
	--type $(ENT_SEARCH_INDEX) $(ENT_ARGS)

	@python scripts/build_kg_index.py \
	data/kg-index/freebase/properties \
	data/kg-index/freebase/properties/$(PROP_SEARCH_INDEX) \
	--type $(PROP_SEARCH_INDEX) $(PROP_ARGS)

dbpedia-kg-data:
	# dbpedia entities
	# https://qlever.cs.uni-freiburg.de/dbpedia/T1q23G
	@mkdir -p data/kg-index/dbpedia/entities
	@curl -s $(DBPEDIA_URL) -H "Accept: text/csv" \
	--data-urlencode query="PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX dbo: <http://dbpedia.org/ontology/> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX dbp: <http://dbpedia.org/property/> PREFIX foaf: <http://xmlns.com/foaf/0.1/> SELECT ?label ?score (GROUP_CONCAT(DISTINCT ?alias; SEPARATOR=\";;;\") AS ?synonyms) ?id (GROUP_CONCAT(DISTINCT ?info; SEPARATOR=\";;;\") AS ?infos) WHERE { { SELECT ?id (COUNT(?id) AS ?score) WHERE { ?id ?p ?o } GROUP BY ?id } ?id @en@rdfs:label ?label . OPTIONAL { { ?id @en@dbp:synonyms ?alias } UNION { ?id @en@dbo:alias ?alias } UNION { ?id @en@dbo:alternativeName ?alias } UNION { ?id @en@foaf:nick ?alias } } OPTIONAL { { ?id @en@rdfs:comment ?info } UNION { ?id rdf:type ?type_ . FILTER(STRSTARTS(STR(?type_), \"http://dbpedia.org/ontology/\")) . ?type_ @en@rdfs:label ?type . BIND(CONCAT(\"is a \", ?type) AS ?info) } UNION { ?id rdfs:subClassOf ?sup_ . FILTER(STRSTARTS(STR(?sup_), \"http://dbpedia.org/ontology/\")) . ?sup_ @en@rdfs:label ?sup . BIND(CONCAT(\"is subclass of \", ?sup) AS ?info) } } } GROUP BY ?label ?score ?id ORDER BY DESC(?score)" \
	--data-urlencode timeout=$(QLEVER_TIMEOUT) \
	--data-urlencode access-token=$(QLEVER_ACCESS_TOKEN) \
	| python scripts/prepare_kg_data.py \
	--kg dbpedia \
	> data/kg-index/dbpedia/entities/data.tsv

	@python scripts/build_kg_data_and_mapping.py \
	data/kg-index/dbpedia/entities/data.tsv \
	data/kg-index/dbpedia/entities/offsets.bin \
	data/kg-index/dbpedia/entities/mapping.bin \
	--overwrite

	# dbpedia properties
	# https://qlever.cs.uni-freiburg.de/dbpedia/gNf0hw
	@mkdir -p data/kg-index/dbpedia/properties
	@curl -s $(DBPEDIA_URL) -H "Accept: text/csv" \
	--data-urlencode query="PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX dbo: <http://dbpedia.org/ontology/> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX dbp: <http://dbpedia.org/property/> PREFIX foaf: <http://xmlns.com/foaf/0.1/> SELECT ?label ?score (GROUP_CONCAT(DISTINCT ?alias; SEPARATOR=\";;;\") AS ?synonyms) ?id (GROUP_CONCAT(DISTINCT ?info; SEPARATOR=\";;;\") AS ?infos) WHERE { { SELECT ?id (COUNT(?id) AS ?score) WHERE { ?s ?id ?o } GROUP BY ?id } ?id @en@rdfs:label ?label . ?id rdf:type rdf:Property . BIND(\"\" AS ?alias) OPTIONAL { { ?id @en@rdfs:comment ?info } UNION { ?id rdfs:subPropertyOf/@en@rdfs:label ?sup . BIND(CONCAT(\"is subproperty of \", ?sup) AS ?info) } UNION { ?id rdfs:range/@en@rdfs:label ?range . BIND(CONCAT(\"has range \", ?range) AS ?info) } UNION { ?id rdfs:domain/@en@rdfs:label ?domain . BIND(CONCAT(\"has domain \", ?domain) AS ?info) } } } GROUP BY ?label ?score ?id ORDER BY DESC(?score)" \
	--data-urlencode timeout=$(QLEVER_TIMEOUT) \
	--data-urlencode access-token=$(QLEVER_ACCESS_TOKEN) \
	| python scripts/prepare_kg_data.py \
	--kg dbpedia \
	--add-id-as-synonym \
	> data/kg-index/dbpedia/properties/data.tsv

	@python scripts/build_kg_data_and_mapping.py \
	data/kg-index/dbpedia/properties/data.tsv \
	data/kg-index/dbpedia/properties/offsets.bin \
	data/kg-index/dbpedia/properties/mapping.bin \
	--overwrite

dbpedia-kg-indices:
	@python scripts/build_kg_index.py \
	data/kg-index/dbpedia/entities \
	data/kg-index/dbpedia/entities/$(ENT_SEARCH_INDEX) \
	--type $(ENT_SEARCH_INDEX) $(ENT_ARGS)

	@python scripts/build_kg_index.py \
	data/kg-index/dbpedia/properties \
	data/kg-index/dbpedia/properties/$(PROP_SEARCH_INDEX) \
	--type $(PROP_SEARCH_INDEX) $(PROP_ARGS)

dblp-kg-data:
	# dblp entities
	# https://qlever.cs.uni-freiburg.de/dblp/2quXWZ
	@mkdir -p data/kg-index/dblp/entities
	@curl -s $(DBLP_URL) -H "Accept: text/csv" \
	--data-urlencode query="PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX dblp: <https://dblp.org/rdf/schema#> SELECT ?label ?score (GROUP_CONCAT(DISTINCT ?alias; SEPARATOR=\";;;\") AS ?synonyms) ?id (GROUP_CONCAT(DISTINCT ?info; SEPARATOR=\";;;\") AS ?infos) WHERE { { SELECT ?id (COUNT(*) AS ?score) WHERE { ?id rdfs:label [] . ?id ?p ?o } GROUP BY ?id } ?id rdfs:label ?label . OPTIONAL { ?id dblp:primaryCreatorName ?alias } OPTIONAL { ?id rdfs:comment ?info } } GROUP BY ?label ?score ?id ORDER BY DESC(?score)" \
	--data-urlencode timeout=$(QLEVER_TIMEOUT) \
	--data-urlencode access-token=$(QLEVER_ACCESS_TOKEN) \
	| python scripts/prepare_kg_data.py \
	--kg dblp \
	> data/kg-index/dblp/entities/data.tsv

	@python scripts/build_kg_data_and_mapping.py \
	data/kg-index/dblp/entities/data.tsv \
	data/kg-index/dblp/entities/offsets.bin \
	data/kg-index/dblp/entities/mapping.bin \
	--overwrite

	# dblp properties
	# https://qlever.cs.uni-freiburg.de/dblp/OBzYPV
	@mkdir -p data/kg-index/dblp/properties
	@curl -s $(DBLP_URL) -H "Accept: text/csv" \
	--data-urlencode query="PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> SELECT ?label ?score (GROUP_CONCAT(DISTINCT ?alias; SEPARATOR=\";;;\") AS ?synonyms) ?id (GROUP_CONCAT(DISTINCT ?info; SEPARATOR=\";;;\") AS ?infos) WHERE { { SELECT ?id (COUNT(?id) AS ?score) WHERE { ?s ?id ?o } GROUP BY ?id } ?id rdf:type rdf:Property . BIND(\"\" AS ?label) BIND(\"\" AS ?alias) OPTIONAL { { ?id rdfs:comment ?info } UNION { ?id rdfs:subPropertyOf ?type_ . ?type_ rdfs:label ?info } UNION { ?id rdfs:range ?range_ . ?range_ rdfs:label ?info } UNION { ?id rdfs:domain ?domain_ . ?domain_ rdfs:label ?info } } } GROUP BY ?label ?score ?id ORDER BY DESC(?score)" \
	--data-urlencode timeout=$(QLEVER_TIMEOUT) \
	--data-urlencode access-token=$(QLEVER_ACCESS_TOKEN) \
	| python scripts/prepare_kg_data.py \
	--kg dblp \
	--add-id-as-synonym \
	> data/kg-index/dblp/properties/data.tsv

	@python scripts/build_kg_data_and_mapping.py \
	data/kg-index/dblp/properties/data.tsv \
	data/kg-index/dblp/properties/offsets.bin \
	data/kg-index/dblp/properties/mapping.bin \
	--overwrite

dblp-kg-indices:
	@python scripts/build_kg_index.py \
	data/kg-index/dblp/entities \
	data/kg-index/dblp/entities/$(ENT_SEARCH_INDEX) \
	--type $(ENT_SEARCH_INDEX) $(ENT_ARGS)

	@python scripts/build_kg_index.py \
	data/kg-index/dblp/properties \
	data/kg-index/dblp/properties/$(PROP_SEARCH_INDEX) \
	--type $(PROP_SEARCH_INDEX) $(PROP_ARGS)

ORKG_VERSION=orkg

orkg-kg-data:
	# orkg entities
	# https://qlever.cs.uni-freiburg.de/orkg/AaYKTn
	@mkdir -p data/kg-index/$(ORKG_VERSION)/entities
	@curl -s $(ORKG_URL) -H "Accept: text/csv" \
	--data-urlencode query="PREFIX orkgp: <http://orkg.org/orkg/predicate/> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> SELECT ?label ?score (GROUP_CONCAT(DISTINCT ?alias; SEPARATOR=\";;;\") AS ?synonyms) ?id (GROUP_CONCAT(DISTINCT ?info; SEPARATOR=\";;;\") AS ?infos) WHERE { { SELECT ?id ?label (COUNT(?id) AS ?score) WHERE { ?id rdfs:label ?label . ?id ?p ?o } GROUP BY ?id ?label } BIND(\"\" AS ?alias) OPTIONAL { { ?id rdf:type/rdfs:label ?type_label. BIND(CONCAT(\"has type \", ?type_label) AS ?info) } UNION { ?id orkgp:description ?info } } } GROUP BY ?label ?score ?id ORDER BY DESC(?score)" \
	--data-urlencode timeout=$(QLEVER_TIMEOUT) \
	--data-urlencode access-token=$(QLEVER_ACCESS_TOKEN) \
	| python scripts/prepare_kg_data.py \
	--kg orkg \
	> data/kg-index/$(ORKG_VERSION)/entities/data.tsv

	@python scripts/build_kg_data_and_mapping.py \
	data/kg-index/$(ORKG_VERSION)/entities/data.tsv \
	data/kg-index/$(ORKG_VERSION)/entities/offsets.bin \
	data/kg-index/$(ORKG_VERSION)/entities/mapping.bin \
	--overwrite

	# orkg properties
	# https://qlever.cs.uni-freiburg.de/orkg/GqLOQH
	@mkdir -p data/kg-index/$(ORKG_VERSION)/properties
	@curl -s $(ORKG_URL) -H "Accept: text/csv" \
	--data-urlencode query="PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX orkgc: <http://orkg.org/orkg/class/> SELECT ?label ?score (GROUP_CONCAT(DISTINCT ?alias; SEPARATOR=\";;;\") AS ?synonyms) ?id (GROUP_CONCAT(DISTINCT ?info; SEPARATOR=\";;;\") AS ?infos) WHERE { { SELECT ?id (COUNT(?id) AS ?score) WHERE { ?s ?id ?o } GROUP BY ?id } ?id rdfs:label ?label . ?id rdf:type orkgc:Predicate . BIND(\"\" AS ?alias) BIND(\"\" AS ?info) } GROUP BY ?label ?score ?id ORDER BY DESC(?score)" \
	--data-urlencode timeout=$(QLEVER_TIMEOUT) \
	--data-urlencode access-token=$(QLEVER_ACCESS_TOKEN) \
	| python scripts/prepare_kg_data.py \
	--kg orkg \
	--add-id-as-synonym \
	> data/kg-index/$(ORKG_VERSION)/properties/data.tsv

	@python scripts/build_kg_data_and_mapping.py \
	data/kg-index/$(ORKG_VERSION)/properties/data.tsv \
	data/kg-index/$(ORKG_VERSION)/properties/offsets.bin \
	data/kg-index/$(ORKG_VERSION)/properties/mapping.bin \
	--overwrite

orkg-kg-indices:
	@python scripts/build_kg_index.py \
	data/kg-index/$(ORKG_VERSION)/entities \
	data/kg-index/$(ORKG_VERSION)/entities/$(ENT_SEARCH_INDEX) \
	--type $(ENT_SEARCH_INDEX) $(ENT_ARGS)

	@python scripts/build_kg_index.py \
	data/kg-index/$(ORKG_VERSION)/properties \
	data/kg-index/$(ORKG_VERSION)/properties/$(PROP_SEARCH_INDEX) \
	--type $(PROP_SEARCH_INDEX) $(PROP_ARGS)
