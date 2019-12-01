namespace GraphParallelAlgorithms
{
    internal class Program
    {
        private static void Main()
        {
            var g = new Graph("input.txt");

            g.WriteFloyd();
            g.WriteKruskal();
            g.WritePrim();
        }
    }
}