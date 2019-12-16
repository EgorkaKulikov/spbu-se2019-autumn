using System.Threading;

namespace Task05
{
    public class Node
    {
        public int Value { get; set; }
        public Node Left { get; set; }
        public Node Right { get; set; }
        public readonly Mutex Mtx = new Mutex();
    }
}