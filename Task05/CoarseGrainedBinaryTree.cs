using System.Threading;

namespace Task05
{
    public class CoarseGrainedBinaryTree
    {
        public BinaryTreeNode Root;
        private readonly Mutex _mutex = new Mutex();
        public bool Find(int key)
        {
            bool result;
            try
            {
                _mutex.WaitOne();
                result = FindNode(Root, key) != null;
            }
            finally
            {
                _mutex.ReleaseMutex();
            }
            return result;
        }

        public void Insert(int key)
        {
            try
            {
                _mutex.WaitOne();
                InsertNode(new BinaryTreeNode(key));
            }
            finally
            {
                _mutex.ReleaseMutex();
            }
        }
        
        private BinaryTreeNode FindNode(BinaryTreeNode node, int key)
        {
            while (true)
            {
                if (node == null || node.Key == key)
                {
                    return node;
                }

                node = node.Key > key ? node.Left : node.Right;
            }
        }
        
        private void InsertNode(BinaryTreeNode node)
        {
            if (Root == null)
            {
                Root = node;
            }
            else if (FindNode(Root, node.Key) == null)
            {
                var currentNode = Root;

                while (currentNode != null)
                {
                    if (node.Key > currentNode.Key)
                    {
                        if (currentNode.Right == null)
                        {
                            node.Parent = currentNode;
                            currentNode.Right = node;
                            
                            break;
                        }

                        currentNode = currentNode.Right;
                    }
                    else
                    {
                        if (currentNode.Left == null)
                        {
                            node.Parent = currentNode;
                            currentNode.Left = node;
                            break;
                        }

                        currentNode = currentNode.Left;
                    }
                }
            }
        }
        
    }
}