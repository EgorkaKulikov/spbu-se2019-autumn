namespace Concurrent_trees.Fine
{
    public class Tree
    {
        private Node root;

        public bool Find(int key)
        {
            lock (this)
            {
                return root != null && FindNode(ref root, ref root, key);
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

                InsertNode(ref root, ref root, ref newNode);
            }
        }

        private bool FindNode(ref Node node, ref Node parent, int key)
        {
            lock (parent)
            lock (node)
            {
                if (node.Key == key) return true;

                var next = node.Key > key ? node.Left : node.Right;

                return next != null && FindNode(ref next, ref node, key);
            }
        }

        private void InsertNode(ref Node node, ref Node parent, ref Node newNode)
        {
            lock (parent)
            lock (node)
            {
                if (node.Key > newNode.Key)
                {
                    if (node.Left == null)
                    {
                        node.Left = newNode;
                        node.Left.Parent = node;
                        return;
                    }

                    InsertNode(ref node.Left, ref node, ref newNode);
                }
                else
                {
                    if (node.Right == null)
                    {
                        node.Right = newNode;
                        node.Right.Parent = node;
                        return;
                    }

                    InsertNode(ref node.Right, ref node, ref newNode);
                }
            }
        }
    }
}