namespace Concurrent_trees.Coarse
{
    public class Tree
    {
        private Node root;

        public bool Find(int key)
        {
            if (root == null) return false;

            lock (this)
            {
                var node = root;

                while (node != null)
                {
                    if (node.Key == key) return true;

                    node = node.Key > key ? node.Left : node.Right;
                }

                return false;
            }
        }

        public void Insert(int key)
        {
            lock (this)
            {
                if (Find(key)) return;

                var newNode = new Node(key);

                if (root == null)
                {
                    root = newNode;
                    return;
                }

                var node = root;

                while (true)
                {
                    if (node.Key > key)
                    {
                        if (node.Left == null)
                        {
                            node.Left = newNode;
                            node.Left.Parent = node;
                            return;
                        }

                        node = node.Left;
                    }
                    else
                    {
                        if (node.Right == null)
                        {
                            node.Right = newNode;
                            node.Right.Parent = node;
                            return;
                        }

                        node = node.Right;
                    }
                }
            }
        }
    }
}