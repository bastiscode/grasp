export const APP_COLORS = Object.freeze({
  uniBlue: '#344A9A',
  uniDarkBlue: '#000149',
  uniRed: '#C1002A',
  uniGray: '#B4B4B4',
  uniGreen: '#00A082',
  uniYellow: '#BEAA3C',
  uniPink: '#A35394',
  surface: '#FFFFFF'
});

export const BRAND_LINKS = Object.freeze({
  chair: 'https://ad.cs.uni-freiburg.de',
  repo: 'https://github.com/ad-freiburg/grasp',
  methodPaper: 'https://ad-publications.cs.uni-freiburg.de/ISWC_grasp_WB_2025.pdf',
  systemPaper: 'https://ad-publications.cs.uni-freiburg.de/ISWC_grasp_demo_WB_2025.pdf',
  entityLinkingPaper:
    'https://ad-publications.cs.uni-freiburg.de/SEMTAB_entity_linking_grasp_WB_2025.pdf',
  evaluation: 'https://grasp.cs.uni-freiburg.de/evaluate/',
  data: 'https://ad-publications.cs.uni-freiburg.de/grasp/'
});

export const BACKEND_CONFIG = Object.freeze({
  hostAndPort: 'grasp.cs.uni-freiburg.de',
  secure: true,
  baseURL: '/api'
});

export const TASKS = Object.freeze([
  {
    id: 'sparql-qa',
    name: 'SPARQL QA',
    tooltip:
      'Answer questions by generating a corresponding SPARQL query over one or more knowledge graphs.'
  },
  {
    id: 'general-qa',
    name: 'General QA',
    tooltip:
      'Answer questions by retrieving relevant information from knowledge graphs.'
  },
  {
    id: 'cea',
    name: 'Cell Entity Annotation',
    tooltip:
      'Upload a CSV table to annotate each cell with corresponding knowledge graph entities.'
  }
]);

export const QLEVER_HOSTS = Object.freeze([
  'qlever.cs.uni-freiburg.de',
  'qlever.informatik.uni-freiburg.de',
  'qlever.dev'
]);

export const endpointFor = (path) => {
  const protocol = BACKEND_CONFIG.secure ? 'https' : 'http';
  return `${protocol}://${BACKEND_CONFIG.hostAndPort}${BACKEND_CONFIG.baseURL}${path}`;
};

export const wsEndpoint = () => {
  const protocol = BACKEND_CONFIG.secure ? 'wss' : 'ws';
  return `${protocol}://${BACKEND_CONFIG.hostAndPort}${BACKEND_CONFIG.baseURL}/live`;
};

export const configEndpoint = () => endpointFor('/config');
export const kgEndpoint = () => endpointFor('/knowledge_graphs');
export const saveSharedStateEndpoint = () => endpointFor('/save');
export const loadSharedStateEndpoint = (id) => endpointFor(`/load/${encodeURIComponent(id)}`);
