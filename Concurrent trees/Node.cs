namespace Concurrent_trees
{
    public class Node
    {
        public int Key;
        public Node Left;
        public Node Parent;
        public Node Right;

        public Node(int key) => Key = key;
    }
}