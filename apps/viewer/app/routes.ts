import { type RouteConfig, index, route } from "@react-router/dev/routes";

export default [
  index("routes/rollouts.tsx"),
  route("rollouts/:task/:id", "routes/rollout.tsx"),
  route("compare", "routes/compare.tsx"),
] satisfies RouteConfig;
