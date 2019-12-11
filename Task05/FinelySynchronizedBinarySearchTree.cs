using System;

namespace Task05
{
    class FinelySynchronizedBinarySearchTree<K, V> : Tree<K, V, BinarySearchNode<K, V>>
        where K : IComparable<K>, new()
        where V : struct
    {
        public FinelySynchronizedBinarySearchTree(params (K, V)[] elements)
        {
            overRoot = new BinarySearchNode<K, V>(new K(), new V());
            insert(elements);
        }

        protected override BinarySearchNode<K, V> createRoot(K key, V value) => new BinarySearchNode<K, V>(key, value);
        
        protected override NodeStatus setWithInfo(K key, V? value)
        {
            overRoot.mutex.WaitOne();
            if (root == null)
            {
                if (value == null)
                {
                    overRoot.mutex.ReleaseMutex();
                    return NodeStatus.NodeNotFounded;
                }
                else
                {
                    root = createRoot(key, value.Value);
                    overRoot.mutex.ReleaseMutex();
                    return NodeStatus.NodeCreate;
                }
            }
            root.mutex.WaitOne();
            BinarySearchNode<K, V> node = root, nextNode = root.nextNode(key), prevNode = overRoot;
            while (nextNode != null)
            {
                prevNode.mutex.ReleaseMutex();
                nextNode.mutex.WaitOne();
                prevNode = node;
                node = nextNode;
                nextNode = nextNode.nextNode(key);
            }
            if (value != null)
            {
                if (node.key.CompareTo(key) == 0)
                {
                    node.value = value.Value;
                    prevNode.mutex.ReleaseMutex();
                    node.mutex.ReleaseMutex();
                    return NodeStatus.NodeUpdate;
                }
                node.createSon(key, value.Value, node.findKeyType(key));
                prevNode.mutex.ReleaseMutex();
                node.mutex.ReleaseMutex();
                return NodeStatus.NodeCreate;
            }
            if (node.key.CompareTo(key) != 0)
            {
                prevNode.mutex.ReleaseMutex();
                node.mutex.ReleaseMutex();
                return NodeStatus.NodeNotFounded;
            }
            if (node.left == null && node.right == null)
            {
                if (node.type == SonType.Root)
                    root = null;
                else
                    prevNode.setSon(null, node.type);
                prevNode.mutex.ReleaseMutex();
                node.mutex.ReleaseMutex();
                return NodeStatus.NodeDelete;
            }
            BinarySearchNode<K, V> nodeWithNextKey = node.right.left, nodeRightSon = node.right, nodeLeftSon = node.left;
            if (node.right == null)
            {
                nodeLeftSon.mutex.WaitOne();
                if (node.type == SonType.Root)
                {
                    root = nodeLeftSon;
                    nodeLeftSon.parent = null;
                    node.left = null;
                }
                else
                    nodeLeftSon.setFather(prevNode, node.type);
                nodeLeftSon.mutex.ReleaseMutex();
                prevNode.mutex.ReleaseMutex();
                node.mutex.ReleaseMutex();
                return NodeStatus.NodeDelete;
            }
            nodeRightSon.mutex.WaitOne();
            if (nodeRightSon.left == null)
            {
                if (node.type == SonType.Root)
                    root = nodeRightSon;
                if (nodeLeftSon != null)
                {
                    nodeLeftSon.mutex.WaitOne();
                }
                nodeRightSon.setSon(nodeLeftSon, SonType.LeftSon);
                nodeRightSon.setFather(node.parent, node.type);
                if (nodeLeftSon != null)
                {
                    nodeLeftSon.mutex.ReleaseMutex();
                }
                nodeRightSon.mutex.ReleaseMutex();
                prevNode.mutex.ReleaseMutex();
                node.mutex.ReleaseMutex();
                return NodeStatus.NodeDelete;
            }
            nodeWithNextKey.mutex.WaitOne();
            while (nodeWithNextKey.left != null)
            {
                nodeWithNextKey.parent.mutex.ReleaseMutex();
                nodeWithNextKey.left.mutex.WaitOne();
                nodeWithNextKey = nodeWithNextKey.left;
            }
            BinarySearchNode<K, V> nodeWithNextKeyRightSon = nodeWithNextKey.right, nodeWithNextKeyParent = nodeWithNextKey.parent;
            if (node.type == SonType.Root)
                root = nodeWithNextKey;
            if (nodeWithNextKeyRightSon != null)
            {
                nodeWithNextKeyRightSon.mutex.WaitOne();
            }
            nodeWithNextKeyParent.setSon(nodeWithNextKeyRightSon, SonType.LeftSon);
            if (nodeWithNextKeyRightSon != null)
            {
                nodeWithNextKeyRightSon.mutex.ReleaseMutex();
            }
            nodeWithNextKeyParent.mutex.ReleaseMutex();
            nodeRightSon.mutex.WaitOne();
            if (nodeLeftSon != null)
            {
                nodeLeftSon.mutex.WaitOne();
            }
            nodeWithNextKey.moveOn(node);
            nodeWithNextKey.mutex.ReleaseMutex();
            if (nodeLeftSon != null)
            {
                nodeLeftSon.mutex.ReleaseMutex();
            }
            nodeRightSon.mutex.ReleaseMutex();
            prevNode.mutex.ReleaseMutex();
            node.mutex.ReleaseMutex();
            return NodeStatus.NodeDelete;
        }
    }
}
