export function load({ params }) {
  return {
    loadId: params.id ?? null
  };
}

export const prerender = false;
