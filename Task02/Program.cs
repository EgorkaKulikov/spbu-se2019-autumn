namespace Task02
{
    class Program
    {
        static void Main()
        {
            var graph = new Graph("input.txt");
            graph.PrintFloyd();
            graph.PrintKruskal(); 
            graph.PrintPrim();
        }
    }
}