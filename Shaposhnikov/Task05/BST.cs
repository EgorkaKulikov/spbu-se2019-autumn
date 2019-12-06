using System;
using System.Diagnostics;
using System.Threading;

namespace Task05
{
    public class Tree
    {
        public Node Root { get; private set; }

        private readonly Mutex _coarseMtx = new Mutex();
        
        private readonly Mutex _rootEmpty = new Mutex();

        public void CoarseInsert(int value)
        {
            Node prev = null, current = Root;
            _coarseMtx.WaitOne();

            try
            {
                while (current != null)
                {
                    prev = current;
                    if (value < current.Value)
                        current = current.Left;
                    else if (value > current.Value)
                        current = current.Right;
                    else
                        return; //already exists
                }

                var newNode = new Node {Value = value};

                if (Root == null)
                    Root = newNode;
                else
                {
                    Debug.Assert(prev != null, nameof(prev) + " != null");
                    if (value < prev.Value)
                        prev.Left = newNode;
                    else
                        prev.Right = newNode;
                }
            }
            finally
            {
                _coarseMtx.ReleaseMutex();
            }
        }
        
        public void FineInsert(int value)
        {
            Node prev = null, current = Root;

            if (current != null)
                current.Mtx.WaitOne();
            else
            {
                _rootEmpty.WaitOne();
                if (Root != null)
                {
                    current = Root;
                    _rootEmpty.ReleaseMutex();
                    current.Mtx.WaitOne();
                }
            }
            
            while (current != null)
            {
                prev?.Mtx.ReleaseMutex();

                prev = current;
                if (value < current.Value)
                {
                    current = current.Left;
                    current?.Mtx.WaitOne();
                }
                else if (value > current.Value)
                {
                    current = current.Right;
                    current?.Mtx.WaitOne();
                }
                else
                { 
                    current.Mtx.ReleaseMutex();
                    return; //already exists
                }
            }

            var newNode = new Node {Value = value};

            if (Root == null)
            {
                Root = newNode;
                _rootEmpty.ReleaseMutex();
            }
            else
            {
                Debug.Assert(prev != null, nameof(prev) + " != null");
                if (value < prev.Value)
                {
                    prev.Left = newNode;
                    prev.Mtx.ReleaseMutex();
                }
                else 
                {
                    prev.Right = newNode;
                    prev.Mtx.ReleaseMutex();
                }
            }
        }
        
        public Node CoarseFind(int value)
        {
            _coarseMtx.WaitOne();
            try
            {
                return CoarseFind(value, Root);
            }
            finally
            {
                _coarseMtx.ReleaseMutex();
            }
        }
        
        public Node FineFind(int key)
        {
            if (Root == null) return null;
            
            Root.Mtx.WaitOne();
            return FineFind(key, Root);
        }

        public void CoarseRemove(int key)
        {
            _coarseMtx.WaitOne();
            
            try
            {
                CoarseRemove(Root, key);
            }
            finally
            {
                if (Root != null)
                    _coarseMtx.ReleaseMutex();
            }
        }
        
        public void FineRemove(int key)
        {
            if (Root == null) return;
            
            Root.Mtx.WaitOne();
            if (Root.Value == key)
            {
                var old = Root;
                
                if (Root.Left == null)
                {
                    Root = Root.Right;
                    old.Mtx.ReleaseMutex();
                }
                else if(Root.Right == null)
                {
                    Root = Root.Left;
                    old.Mtx.ReleaseMutex();
                }
                else
                {
                    var right = Root.Right;
                    right.Mtx.WaitOne();
                    
                    var min = FineMinValue(right);
                    Root.Value = min;
                    
                    if (min != right.Value)
                    {
                        old.Mtx.ReleaseMutex();
                        FineRemove(right, min);
                    }
                    else
                    {
                        Root.Right = right.Right;
                        right.Mtx.ReleaseMutex();
                        Root.Mtx.ReleaseMutex();
                    }
                }
            }
            else 
                FineRemove(Root, key);
        }

        private static Node CoarseFind(int value, Node parent)
        {
            while (true)
            {
                if (parent != null)
                {
                    //nvm it's an IOCCC practice 
                    if (value == parent.Value) return parent;
                    var value1 = value;
                    parent = value1 < parent.Value
                        ? parent.Left
                        : parent.Right;
                    continue;
                }

                return null;
            }
        }

        private static Node FineFind(int key, Node parent)
        {
            while (true)
            {
                if (parent == null) return null;

                if (key == parent.Value)
                {
                    parent.Mtx.ReleaseMutex();
                    return parent;
                }

                if (key < parent.Value)
                {
                    var left = parent.Left;
                    left?.Mtx.WaitOne();
                    parent.Mtx.ReleaseMutex();

                    return FineFind(key, left);
                }
                else
                {
                    var right = parent.Right; 
                    right?.Mtx.WaitOne();
                    parent.Mtx.ReleaseMutex();
                    
                    return FineFind(key, right);
                }
            }
        }

        private static Node CoarseRemove(Node parent, int key)
        {
            if (parent == null) return null;

            //laddering the tree for sought-for node
            if (key < parent.Value)
                parent.Left = CoarseRemove(parent.Left, key);
            else if (key > parent.Value)
                parent.Right = CoarseRemove(parent.Right, key);
            //the remaining case is parent's value equals key
            else
            {
                //case node with one or zero children
                if (parent.Right == null)
                    return parent.Left;
                if (parent.Left == null)
                    return parent.Right;

                // get the next lowest value after parent's
                parent.Value = MinValue(parent.Right);

                // and then delete the node with this lowest key
                parent.Right = CoarseRemove(parent.Right, parent.Value);
            }

            return parent;
        }
        
        private static void FineRemove(Node current, int key)
        {
            var parent = current;
            while (true)
            {
                if (current.Value == key)
                {
                    if (current.Left == null)
                    {
                        if (key < parent.Value)
                            parent.Left = current.Right;
                        else
                            parent.Right = current.Right;
                            
                        parent.Mtx.ReleaseMutex();
                        current.Mtx.ReleaseMutex();

                        return;
                    }
                    else if(current.Right == null)
                    {
                        if (key < parent.Value)
                            parent.Left = current.Left;
                        else
                            parent.Right = current.Left;
                            
                        parent.Mtx.ReleaseMutex();
                        current.Mtx.ReleaseMutex();

                        return;
                    }
                    else
                    {
                        var right = current.Right;
                        right.Mtx.WaitOne();
                        
                        var min = FineMinValue(right);
                        current.Value = min;
                    
                        if (min != right.Value)
                        {
                            current.Mtx.ReleaseMutex();
                            parent.Mtx.ReleaseMutex();
                            FineRemove(right, min);
                            return;
                        }
                        else
                        {
                            current.Right = right.Right;
                            right.Mtx.ReleaseMutex();
                            current.Mtx.ReleaseMutex();
                            parent.Mtx.ReleaseMutex();
                            return;
                        }
                    }
                }
                
                if (parent != current)
                {
                    parent.Mtx.ReleaseMutex();
                    parent = current;
                }

                if (key < current.Value)
                {
                    current = current.Left;
                    if (current == null)
                    {
                        parent.Mtx.ReleaseMutex();
                        return;
                    }

                    current.Mtx.WaitOne();
                }
                else if (key > current.Value)
                {
                    current = current.Right;
                    if (current == null)
                    {
                        parent.Mtx.ReleaseMutex();
                        return;
                    }

                    current.Mtx.WaitOne();
                }
            }
        }
        

        private static int MinValue(Node node)
        {
            var min = node.Value;

            while (node.Left != null)
            {
                min = node.Left.Value;
                node = node.Left;
            }

            return min;
        }
        
        private static int FineMinValue(Node node)
        {
            var min = node.Value;

            var left = node.Left;
            if (left == null)
                return min;
            left.Mtx.WaitOne();
            
            while (left != null)
            {
                min = left.Value;
                var prev = left;
                
                left = left.Left;
                left?.Mtx.WaitOne();

                prev.Mtx.ReleaseMutex();
            }

            return min;
        }
        
        // probably for unit-testing
        public int GetDepth()
        {
            return GetDepth(Root);
        }

        private static int GetDepth(Node parent)
        {
            return parent == null ? 0
                : Math.Max(GetDepth(parent.Left), GetDepth(parent.Right)) + 1;
        }
    }
}