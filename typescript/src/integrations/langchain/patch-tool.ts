// import { tool as originalTool } from "@langchain/core/tools";

// import { getCurrentSpan } from "../../tracing/tracing";
// import { BaseMetric } from "../../metrics/base-metrics";

// type AnyFunc = (...args: any[]) => any;

// export function tool(...args: any[]) {
//   let metrics: BaseMetric[] | undefined = undefined;
//   let metricCollection: string | undefined = undefined;
//   let kwargs: Record<string, any> = {};

//   if (
//     args.length > 0 &&
//     typeof args[args.length - 1] === "object" &&
//     !Array.isArray(args[args.length - 1])
//   ) {
//     kwargs = { ...args[args.length - 1] };

//     if ("metrics" in kwargs) {
//       metrics = kwargs.metrics;
//       delete kwargs.metrics;
//     }

//     if ("metricCollection" in kwargs || "metric_collection" in kwargs) {
//       metricCollection = kwargs.metricCollection ?? kwargs.metric_collection;
//       delete kwargs.metricCollection;
//       delete kwargs.metric_collection;
//     }

//     args = [...args.slice(0, args.length - 1)];
//   }

//   const toolArgs = args;
//   const toolKwargs = kwargs;

//   function decorator(func: AnyFunc) {
//     const patchedFunc = _patchToolDecorator(func, metrics, metricCollection);

//     return originalTool(...toolArgs, toolKwargs)(patchedFunc);
//   }

//   return decorator;
// }

// function _patchToolDecorator(
//   func: AnyFunc,
//   metrics?: BaseMetric[],
//   metricCollection?: string,
// ): AnyFunc {
//   function wrapper(this: any, ...args: any[]) {
//     const currentSpan = getCurrentSpan();
//     if (currentSpan) {
//       (currentSpan as any).metrics = metrics;
//       (currentSpan as any).metricCollection = metricCollection;
//     }

//     return func.apply(this, args);
//   }

//   try {
//     Object.defineProperty(wrapper, "length", { value: func.length });
//     Object.defineProperty(wrapper, "name", { value: func.name });
//   } catch {
//     // Ignore if not allowed
//   }

//   return wrapper;
// }
