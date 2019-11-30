public partial class Program
{
  private struct Edge
  {
    public int from, to, weight;

    public Edge(int f, int t, int w)
    {
      weight = w;
      to     = t;
      from   = f;
    }
  }
  private static void swap(ref Edge first, ref Edge second)
  {
    int buf = first.weight;
    first.weight = second.weight;
    second.weight = buf;

    buf = first.from;
    first.from = second.from;
    second.from = buf;

    buf = first.to;
    first.to = second.to;
    second.to = buf;
  }
}
