namespace Task02
{
  public partial class Program
  {
    private struct Edge
    {
      public int from, to, weight;

      public Edge(int from, int to, int weight)
      {
        this.from   = from;
        this.to     = to;
        this.weight = weight;
      }
    }
    private static void swap(ref Edge first, ref Edge second)
    {
      var buf = second;
      second  = first;
      first   = buf;
    }
  }
}
