namespace Graphs
{
    public static class Constants
    {
        public static int MinNodes { get; } = 5000;
        public static int MaxNodes { get; } = 7500;
        public static int MinEdges { get; } = 100000;
        public static int MinEdgeValue { get; } = 1;
        public static int MaxEdgeValue { get; } = 999;
        public static int Inf { get; } = MaxEdgeValue + 1;

        public const string NotTheSameResultLog = "Parallel and sequential algorithms have different result";
    }
}