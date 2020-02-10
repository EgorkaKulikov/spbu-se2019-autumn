using System;
using System.Threading;

namespace Task05
{
    public class FineSynchronizedBinarySearchTree<K, V> : BinaryTree<K, V, LockedBinarySearchNode<K, V>>, ILocked
        where K : IComparable<K>
        where V : struct
    {
        protected Mutex Mutex = new Mutex();

        public void Lock()
        {
            Mutex.WaitOne();
        }

        public void Unlock()
        {
            Mutex.ReleaseMutex();
        }

        public FineSynchronizedBinarySearchTree(params (K, V)[] elements)
        {
            Insert(elements);
        }

        public override V? Get(K key)
        {
            Lock();
            if (Root == null)
            {
                Unlock();
                return null;
            }
            Root.Lock();
            BinaryNode<K, V> node = Root,
                nextNode = Root.GetNextNode(key),
                prevNode = null;
            while (nextNode != null)
            {
                (prevNode == null ? this : prevNode as ILocked).Unlock();
                (nextNode as ILocked).Lock();
                prevNode = node;
                node = nextNode;
                nextNode = nextNode.GetNextNode(key);
            }
            if (node.FindKey(key) == Location.ThisNode)
            {
                (prevNode == null ? this : prevNode as ILocked).Unlock();
                (node as ILocked).Unlock();
                return node.Value;
            }
            (prevNode == null ? this : prevNode as ILocked).Unlock();
            (node as ILocked).Unlock();
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
                    Root = new LockedBinarySearchNode<K, V>(key, value.Value);
                    Unlock();
                    return NodeStatus.NodeCreate;
                }
            }
            Root.Lock();
            BinaryNode<K, V> node = Root,
                nextNode = Root.GetNextNode(key),
                prevNode = null;
            while (nextNode != null)
            {
                (prevNode == null ? this : prevNode as ILocked).Unlock();
                (nextNode as ILocked).Lock();
                prevNode = node;
                node = nextNode;
                nextNode = nextNode.GetNextNode(key);
            }
            if (value != null && node.FindKey(key) == Location.ThisNode)
            {
                node.Value = value.Value;
                (prevNode == null ? this : prevNode as ILocked).Unlock();
                (node as ILocked).Unlock();
                return NodeStatus.NodeUpdate;
            }
            if (value != null)
            {
                (node as LockedBinarySearchNode<K, V>)
                    .CreateSon(key, value.Value, node.FindKey(key));
                (prevNode == null ? this : prevNode as ILocked).Unlock();
                (node as ILocked).Unlock();
                return NodeStatus.NodeCreate;
            }
            if (node.FindKey(key) != Location.ThisNode)
            {
                (prevNode == null ? this : prevNode as ILocked).Unlock();
                (node as ILocked).Unlock();
                return NodeStatus.NodeNotFounded;
            }
            if (node.Left == null && node.Right == null)
            {
                if (node.Type == SonType.Root)
                    Root = null;
                prevNode?.SetSon(null, node.Type);
                (prevNode == null ? this : prevNode as ILocked).Unlock();
                (node as ILocked).Unlock();
                return NodeStatus.NodeDelete;
            }
            BinaryNode<K, V> nodeRight = node.Right,
                nodeLeft = node.Left;
            if (nodeRight == null)
            {
                (nodeLeft as ILocked).Lock();
                if (node.Type == SonType.Root)
                    Root = nodeLeft as LockedBinarySearchNode<K, V>;
                nodeLeft.SetFather(prevNode, node.Type);
                (nodeLeft as ILocked).Unlock();
                (prevNode == null ? this : prevNode as ILocked).Unlock();
                (node as ILocked).Unlock();
                return NodeStatus.NodeDelete;
            }
            (nodeRight as ILocked).Lock();
            BinaryNode<K, V> nodeWithNextKey = nodeRight.Left;
            if (nodeWithNextKey == null)
            {
                if (node.Type == SonType.Root)
                    Root = nodeRight as LockedBinarySearchNode<K, V>;
                (nodeLeft as ILocked)?.Lock();
                nodeRight.SetSon(nodeLeft, SonType.LeftSon);
                nodeRight.SetFather(node.Parent, node.Type);
                (nodeLeft as ILocked)?.Unlock();
                (nodeRight as ILocked).Unlock();
                (prevNode == null ? this : prevNode as ILocked).Unlock();
                (node as ILocked).Unlock();
                return NodeStatus.NodeDelete;
            }
            (nodeWithNextKey as ILocked).Lock();
            while (nodeWithNextKey.Left != null)
            {
                (nodeWithNextKey.Parent as ILocked).Unlock();
                (nodeWithNextKey.Left as ILocked).Lock();
                nodeWithNextKey = nodeWithNextKey.Left;
            }
            BinaryNode<K, V> nodeWithNextKeyRight =
                nodeWithNextKey.Right, nodeWithNextKeyParent =
                nodeWithNextKey.Parent;
            if (node.Type == SonType.Root)
                Root = nodeWithNextKey as LockedBinarySearchNode<K, V>;
            (nodeWithNextKeyRight as ILocked)?.Lock();
            nodeWithNextKeyParent.SetSon(nodeWithNextKeyRight,
                SonType.LeftSon);
            (nodeWithNextKeyRight as ILocked)?.Unlock();
            if (nodeWithNextKeyParent != nodeRight)
            {
                (nodeWithNextKeyParent as ILocked).Unlock();
                (nodeRight as ILocked).Lock();
            }
            (nodeLeft as ILocked)?.Lock();
            nodeWithNextKey.MoveOn(node);
            (nodeWithNextKey as ILocked).Unlock();
            (nodeLeft as ILocked)?.Unlock();
            (nodeRight as ILocked).Unlock();
            (prevNode == null ? this : prevNode as ILocked).Unlock();
            (node as ILocked).Unlock();
            return NodeStatus.NodeDelete;
        }
    }
}
