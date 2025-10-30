export const prerender = false;

export function load({ params }) {
  return {
    loadId: params.shareId ?? null
  };
}
