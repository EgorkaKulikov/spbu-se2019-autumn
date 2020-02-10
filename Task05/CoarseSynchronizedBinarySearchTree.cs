using System;
using System.Threading;

namespace Task05
{
    public class CoarseSynchronizedBinarySearchTree<K, V> : BinaryTree<K, V, BinarySearchNode<K, V>>, ILocked
        where K : IComparable<K>
        where V : struct
    {
        protected Mutex Mutex = new Mutex();

        public CoarseSynchronizedBinarySearchTree(params (K, V)[] elements)
        {
            Insert(elements);
        }

        public void Lock()
        {
            Mutex.WaitOne();
        }

        public void Unlock()
        {
            Mutex.ReleaseMutex();
        }

        public override V? Get(K key)
        {
            Lock();
            if (Root == null)
            {
                Unlock();
                return null;
            }
            BinaryNode<K, V> node = Root,
                nextNode = Root.GetNextNode(key),
                prevNode = null;
            while (nextNode != null)
            {
                prevNode = node;
                node = nextNode;
                nextNode = node.GetNextNode(key);
            }
            if (node.FindKey(key) == Location.ThisNode)
            {
                Unlock();
                return node.Value;
            }
            Unlock();
            return null;
        }

        public override NodeStatus SetWithInfo(K key, V? value)
        {
            Lock();
            if (Root == null)
            {
                if (value == null)
                {
                    Unlock();
                    return NodeStatus.NodeNotFounded;
                }
                else
                {
                    Root = new BinarySearchNode<K, V>(key, value.Value);
                    Unlock();
                    return NodeStatus.NodeCreate;
                }
            }
            BinaryNode<K, V> node = Root,
                nextNode = Root.GetNextNode(key),
                prevNode = null;
            while (nextNode != null)
            {
                prevNode = node;
                node = nextNode;
                nextNode = node.GetNextNode(key);
            }
            if (value != null && node.FindKey(key) == Location.ThisNode)
            {
                node.Value = value.Value;
                Unlock();
                return NodeStatus.NodeUpdate;
            }
            if (value != null)
            {
                (node as BinarySearchNode<K, V>)
                    .CreateSon(key, value.Value, node.FindKey(key));
                Unlock();
                return NodeStatus.NodeCreate;
            }
            if (node.FindKey(key) != Location.ThisNode)
            {
                Unlock();
                return NodeStatus.NodeNotFounded;
            }
            if (node.Left == null && node.Right == null)
            {
                if (node.Type == SonType.Root)
                    Root = null;
                prevNode?.SetSon(null, node.Type);
                Unlock();
                return NodeStatus.NodeDelete;
            }
            BinaryNode<K, V> nodeRight = node.Right,
                nodeLeft = node.Left;
            if (nodeRight == null)
            {
                if (node.Type == SonType.Root)
                    Root = nodeLeft as BinarySearchNode<K, V>;
                nodeLeft.SetFather(prevNode, node.Type);
                Unlock();
                return NodeStatus.NodeDelete;
            }
            BinaryNode<K, V> nodeWithNextKey = nodeRight.Left;
            if (nodeWithNextKey == null)
            {
                if (node.Type == SonType.Root)
                    Root = nodeRight as BinarySearchNode<K, V>;
                nodeRight.SetSon(nodeLeft, SonType.LeftSon);
                nodeRight.SetFather(node.Parent, node.Type);
                Unlock();
                return NodeStatus.NodeDelete;
            }
            while (nodeWithNextKey.Left != null)
            {
                nodeWithNextKey = nodeWithNextKey.Left;
            }
            BinaryNode<K, V> nodeWithNextKeyRight =
                nodeWithNextKey.Right, nodeWithNextKeyParent =
                nodeWithNextKey.Parent;
            if (node.Type == SonType.Root)
                Root = nodeWithNextKey as BinarySearchNode<K, V>;
            nodeWithNextKeyParent.SetSon(nodeWithNextKeyRight,
                SonType.LeftSon);
            nodeWithNextKey.MoveOn(node);
            Unlock();
            return NodeStatus.NodeDelete;
        }
    }
}
